# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-segment variant of the PAI dataset.

This module is intentionally decoupled from :mod:`alpamayo.data.pai` so that
upstream changes to ``PAIDataset`` cannot break the multi-segment data path.
The avdi / clip / preprocessor setup that used to come from
``PAIDataset.__init__`` is inlined here.
"""

from typing import Any

import torch
from alpamayo.data.pai_utils_fix import PhysicalAIAVDatasetLocalInterface
from alpamayo_r1.common import logging
from alpamayo_r1.load_physical_aiavdataset_local import load_physical_aiavdataset_local
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class PAIMultiSegmentDataset(Dataset):
    """Dataset that samples multiple segments per clip for data augmentation.

    Instead of using a single fixed t0 per clip, this dataset creates multiple
    training samples by sampling different t0 timestamps within each clip.

    This class is fully self-contained and does **not** inherit from or import
    :class:`alpamayo.data.pai.PAIDataset`. The previous public API (kwargs
    accepted by ``__init__`` and the contract of ``__len__`` / ``__getitem__``)
    is preserved so existing Hydra / Python call sites keep working.
    """

    def __init__(
        self,
        local_dir: str,
        chunk_ids: list[int] | None = None,
        include_extr_intr: bool = False,
        reshape_tensors_for_rl: bool = False,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        num_segments: int = 20,
        segment_start_us: int = 2_000_000,  # 2.0s
        segment_step_us: int = 500_000,  # 0.5s
    ):
        """Initialize the multi-segment dataset.

        Args:
            local_dir: Path to the local directory containing the PAI dataset.
            chunk_ids: List of chunk IDs to load, or a range string (e.g. "0-9").
                If None, all available chunks will be loaded.
            include_extr_intr: Whether to include extrinsics, intrinsics,
                and vehicle dimensions in samples.
            reshape_tensors_for_rl: If True, reshape cameras/timestamps and
                squeeze ego tensors for RL training.
            model_config: Optional config (dict is converted with OmegaConf)
                passed as ``model_config`` into ``hydra.utils.instantiate`` when
                building the preprocessor.
            vla_preprocess_args: If set, Hydra config dict to instantiate a
                preprocessor; its result is stored as ``tokenized_data`` on
                each sample.
            features_metadata: Filename under ``local_dir`` for the features CSV.
            clip_index_metadata: Filename under ``local_dir`` for the clip
                index parquet.
            num_history_steps: History length for ``load_physical_aiavdataset_local``.
            num_future_steps: Future horizon for ``load_physical_aiavdataset_local``.
            time_step: Seconds per step between trajectory samples.
            num_segments: Number of segments (t0 samples) to draw per clip.
            segment_start_us: First t0 (microseconds) for the segment window.
            segment_step_us: Spacing (microseconds) between consecutive t0s
                within a clip.
        """
        # ----- avdi / clip setup (inlined from PAIDataset) -----
        self.avdi = PhysicalAIAVDatasetLocalInterface(
            local_dir=local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
        )
        self.clip_ids = self.avdi.get_all_clip_ids()
        self.include_extr_intr = include_extr_intr
        self.reshape_tensors_for_rl = reshape_tensors_for_rl

        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step

        # ----- optional VLA preprocessor (inlined from PAIDataset) -----
        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(
                vla_preprocess_args, model_config=model_config
            )

        # ----- multi-segment sampling (the part unique to this class) -----
        self.num_segments = num_segments
        self.segment_start_us = segment_start_us
        self.segment_step_us = segment_step_us

        # Pre-compute (clip_id, t0_us) pairs
        future_range_us = int(self.num_future_steps * self.time_step * 1_000_000)
        max_clip_duration_us = 19_900_000  # 19.9s safety margin
        self.clip_segment_t0s: list[tuple[str, int]] = []
        removed_count = 0

        for clip_id in self.clip_ids:
            for seg_idx in range(num_segments):
                t0_us = segment_start_us + seg_idx * segment_step_us
                if t0_us <= max_clip_duration_us - future_range_us:
                    self.clip_segment_t0s.append((clip_id, t0_us))
                else:
                    removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} out-of-bounds segments "
                f"(effective size: {len(self.clip_segment_t0s)})"
            )

    def __len__(self) -> int:
        return len(self.clip_segment_t0s)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        clip_id, t0_us = self.clip_segment_t0s[idx]

        sample_data = load_physical_aiavdataset_local(
            clip_id,
            t0_us=t0_us,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        # squeeze ego motion shape
        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.include_extr_intr:
            sample_data["extr"] = self.avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample_data["intr"] = self.avdi.get_clip_feature(clip_id, "camera_intrinsics")

            vehicle_dimensions = self.avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            sample_data["ego_lwh"] = torch.tensor(
                [vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]
            )
            sample_data["ego_length_offset"] = torch.tensor(
                vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length
            )

        if self.reshape_tensors_for_rl:
            image_frames = sample_data["image_frames"]
            camera_indices = sample_data["camera_indices"]
            absolute_timestamps = sample_data["absolute_timestamps"]
            relative_timestamps = sample_data["relative_timestamps"]

            n_cam, n_frame = image_frames.shape[0], image_frames.shape[1]
            image_frames = image_frames.reshape(n_cam * n_frame, *image_frames.shape[2:]).unsqueeze(1)
            camera_indices = camera_indices.repeat_interleave(n_frame)
            absolute_timestamps = absolute_timestamps.reshape(-1)
            relative_timestamps = relative_timestamps.reshape(-1)

            sample_data["image_frames"] = image_frames
            sample_data["camera_indices"] = camera_indices
            sample_data["absolute_timestamps"] = absolute_timestamps
            sample_data["relative_timestamps"] = relative_timestamps

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
