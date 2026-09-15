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

import io
import json
import os
import pickle
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from physical_ai_av import egomotion, video
from physical_ai_av.dataset import Features
from alpamayo_r1.load_physical_aiavdataset_local import (
    BadLocalZipError,
    load_physical_aiavdataset_local,
)
from alpamayo_r1.common import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class _LocalPAIFeatureReader:
    """Minimal local PAI feature reader, inlined from the retired
    ``pai_utils_fix.PhysicalAIAVDatasetLocalInterface`` so this module stays
    self-contained. Only the surface used by ``COCDataset`` is kept:
    per-clip feature reads (extrinsics / intrinsics / vehicle dimensions).
    """

    def __init__(
        self,
        local_dir: str,
        clip_index_metadata: str = "clip_index.parquet",
        features_metadata: str = "features.csv",
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
    ) -> None:
        self.local_dir = local_dir
        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        features_df = pd.read_csv(
            os.path.join(local_dir, features_metadata), index_col="feature"
        )
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(os.path.join(local_dir, clip_index_metadata))
        self._filter_clips_by_event_t0s()

    def _filter_clips_by_event_t0s(self) -> None:
        """Filter clip_index by event_t0s: keep only events where
        event_t0 >= start_safe_margin_seconds (in us) and
        event_t0 + end_safe_margin_seconds (in us) <= end_timestamp.
        Remove rows whose event_t0s become empty after filtering.
        """
        if "event_t0s" not in self.clip_index.columns:
            return
        start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)
        end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)
        has_end = "end_timestamp" in self.clip_index.columns

        def filter_events(row: pd.Series) -> np.ndarray:
            et0s = row["event_t0s"]
            if et0s is None or (hasattr(et0s, "__len__") and len(et0s) == 0):
                return np.array([], dtype=np.int64)
            arr = np.asarray(et0s, dtype=np.int64)
            mask = arr >= start_margin_us
            if has_end:
                end_ts = int(row["end_timestamp"])
                mask &= (arr + end_margin_us) <= end_ts
            return arr[mask]

        self.clip_index["event_t0s"] = self.clip_index.apply(filter_events, axis=1)
        non_empty = self.clip_index["event_t0s"].apply(lambda x: x is not None and len(x) > 0)
        removed = (~non_empty).sum()
        self.clip_index = self.clip_index.loc[non_empty]
        if removed > 0:
            logger.info(
                "filter_clip_index_by_event_t0s: removed %d rows with empty event_t0s after filtering",
                removed,
            )

    def get_clip_chunk(self, clip_id: str) -> int:
        """Returns the chunk index for `clip_id`."""
        return self.clip_index.at[clip_id, "chunk"]

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        if feature not in self.features.features_df.index:
            logger.warning(
                "Feature %r is not in features_df (available: %s). Returning None.",
                feature,
                list(self.features.features_df.index),
            )
            return None
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_filename, "rb") as f:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(f).loc[clip_id]
            elif chunk_filename.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    elif feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zf.read(clip_files_in_zip["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy(),
                        )
                    else:
                        logger.warning(
                            f"Feature-specific data reader for {feature=} not implemented yet."
                        )
                        return {
                            k: pd.read_parquet(io.BytesIO(zf.read(v)))
                            if v.endswith(".parquet")
                            else io.BytesIO(zf.read(v))
                            for k, v in self.features.get_clip_files_in_zip(
                                clip_id, feature
                            ).items()
                        }
            else:
                raise ValueError(f"Unexpected file extension: {chunk_filename=}.")


class COCDataset(Dataset):
    """Dataset that loads COC reasoning text and joins with PAI trajectory data.

    Each sample in the COC JSONL has a clip_id and ts (timestamp in microseconds).
    The CoC text is injected as the "cot" field.
    """

    def __init__(
        self,
        coc_jsonl_path: str | None = None,
        local_dir: str | None = None,
        include_extr_intr: bool = False,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        clip_index_metadata: str = "clip_index.parquet",
    ):
        if coc_jsonl_path is None:
            coc_jsonl_path = os.environ.get(
                "ALPAMAYO_COC_JSONL", ""
            )
        if local_dir is None:
            local_dir = os.environ.get(
                "ALPAMAYO_DATA_DIR", ""
            )

        assert coc_jsonl_path and os.path.exists(coc_jsonl_path), (
            f"COC JSONL not found: {coc_jsonl_path}. "
            f"Set ALPAMAYO_COC_JSONL env var or pass coc_jsonl_path."
        )
        assert local_dir and os.path.exists(local_dir), (
            f"Data dir not found: {local_dir}. "
            f"Set ALPAMAYO_DATA_DIR env var or pass local_dir."
        )

        logger.info(f"Loading COC annotations from {coc_jsonl_path} ...")
        self.samples: list[dict[str, Any]] = []

        if coc_jsonl_path.endswith(".jsonl"):
            with open(coc_jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.samples.append({
                        "clip_id": row["clip_id"],
                        "ts": row["ts"],
                        "cot": row["coc"],
                    })
        elif coc_jsonl_path.endswith(".json"):
            with open(coc_jsonl_path, "r") as f:
                data = json.load(f)
            for row in data:
                self.samples.append({
                    "clip_id": row["clip_id"],
                    "ts": int(row["ts"]),
                    "cot": row["coc"],
                })
        else:
            raise ValueError(f"Unsupported file format: {coc_jsonl_path}. Expected .jsonl or .json")
        logger.info(f"Loaded {len(self.samples)} COC samples")

        self._cache_dir = os.environ.get("ALPAMAYO_DATA_CACHE_DIR", "data/pai_reasoning_cache")
        self._pkl_cache: dict[str, dict] = {}

        clip_index_path = os.path.join(local_dir, clip_index_metadata)
        if os.path.exists(clip_index_path):
            df = pd.read_parquet(clip_index_path)
            self._local_clip_ids = {
                str(cid) for cid, chunk in zip(df.index, df["chunk"]) if 0 <= chunk <= 49
            }
            # Pre-scan: identify clips whose local camera/egomotion zips are
            # missing or corrupt. These are skipped entirely (no cache fallback,
            # since cache only covers chunk>=50).
            self._bad_local_clip_ids = self._scan_bad_local_clips(df, local_dir)
            if self._bad_local_clip_ids:
                logger.warning(
                    f"Excluding {len(self._bad_local_clip_ids)} local clips whose "
                    f"camera/egomotion zip is missing or corrupt "
                    f"(sample: {sorted(self._bad_local_clip_ids)[:3]})"
                )
        else:
            self._local_clip_ids = set()
            self._bad_local_clip_ids = set()
            logger.warning(f"clip_index not found at {clip_index_path}, all clips will be loaded from cache")

        self.include_extr_intr = include_extr_intr
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.local_dir = local_dir
        self.clip_index_metadata = clip_index_metadata

        # Check that t0 fits within history requirements
        future_range_us = int(num_future_steps * time_step * 1_000_000)
        min_t0_us = int(num_history_steps * time_step * 1_000_000) + 1
        valid_indices = []
        dropped = 0
        for i, s in enumerate(self.samples):
            if s["clip_id"] in self._bad_local_clip_ids:
                dropped += 1
                continue
            t0_us = s["ts"]
            max_future = 20_000_000 - t0_us  # clip max ~20s
            if t0_us > min_t0_us and max_future >= future_range_us:
                valid_indices.append(i)
            else:
                dropped += 1
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} COC samples (t0 out of range or bad local zip) "
                f"(valid: {len(valid_indices)})"
            )
        self.valid_indices = valid_indices

        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def _load_from_cache(self, clip_id: str, t0_us: int) -> dict[str, Any]:
        pkl_path = os.path.join(self._cache_dir, "sample", f"{clip_id}.pkl")

        if clip_id not in self._pkl_cache:
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Cache pkl not found: {pkl_path}")
            with open(pkl_path, "rb") as f:
                self._pkl_cache[clip_id] = pickle.load(f)

        payload = self._pkl_cache[clip_id]
        inner = payload.get(t0_us)
        if inner is None:
            inner = payload.get(str(t0_us))
        if inner is None:
            available = list(payload.keys())[:3]
            raise KeyError(
                f"Timestamp {t0_us} not found in cache for clip {clip_id} "
                f"(available: {available}...)"
            )

        data = inner["data"]

        camera_mode = os.environ.get("ALPAMAYO_CAMERA_MODE", "4cam")
        if camera_mode == "2cam":
            # Cached data is always 4cam order: cross_left(0), front_wide(1), cross_right(2), front_tele(3)
            cam_indices = [1, 3]
            data = dict(data)
            data["image_frames"] = data["image_frames"][cam_indices]
            data["camera_indices"] = data["camera_indices"][cam_indices]
            data["relative_timestamps"] = data["relative_timestamps"][cam_indices]
            data["absolute_timestamps"] = data["absolute_timestamps"][cam_indices]

        return data

    @staticmethod
    def _scan_bad_local_clips(
        clip_index_df: "pd.DataFrame",
        local_dir: str,
    ) -> set[str]:
        """Return the set of clip_ids whose local zips are missing or corrupt.

        For every (chunk in [0,49]) × (camera feature) we open the zip and read
        its central directory. We also probe the egomotion zip. Anything that
        raises (missing, BadZipFile, OSError) marks all clips in that chunk as
        bad — over-filtering a chunk is cheap and safe; under-filtering crashes
        the DataLoader worker.
        """
        import zipfile

        camera_mode = os.environ.get("ALPAMAYO_CAMERA_MODE", "4cam")
        if camera_mode == "2cam":
            cams = ["camera_front_wide_120fov", "camera_front_tele_30fov"]
        else:
            cams = [
                "camera_cross_left_120fov",
                "camera_front_wide_120fov",
                "camera_cross_right_120fov",
                "camera_front_tele_30fov",
            ]

        # Group clip_ids by chunk (only chunks 0..49 are local).
        chunk_to_clips: dict[int, list[str]] = {}
        for cid, chunk in zip(clip_index_df.index, clip_index_df["chunk"]):
            if 0 <= int(chunk) <= 49:
                chunk_to_clips.setdefault(int(chunk), []).append(str(cid))

        bad_clips: set[str] = set()
        for chunk, clips in chunk_to_clips.items():
            chunk_bad = False
            for cam in cams:
                zp = os.path.join(local_dir, "camera", cam, f"{cam}.chunk_{chunk:04d}.zip")
                if not os.path.exists(zp):
                    chunk_bad = True
                    break
                try:
                    with zipfile.ZipFile(zp) as zf:
                        _ = zf.namelist()
                except (zipfile.BadZipFile, OSError):
                    chunk_bad = True
                    break
            if not chunk_bad:
                ego_zp = os.path.join(
                    local_dir, "labels", "egomotion", f"egomotion.chunk_{chunk:04d}.zip"
                )
                if not os.path.exists(ego_zp):
                    chunk_bad = True
                else:
                    try:
                        with zipfile.ZipFile(ego_zp) as zf:
                            _ = zf.namelist()
                    except (zipfile.BadZipFile, OSError):
                        chunk_bad = True
            if chunk_bad:
                bad_clips.update(clips)
        return bad_clips

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.samples[self.valid_indices[idx]]
        clip_id = sample["clip_id"]
        t0_us = sample["ts"]

        if clip_id in self._local_clip_ids:
            try:
                sample_data = load_physical_aiavdataset_local(
                    clip_id,
                    data_dir=self.local_dir,
                    t0_us=t0_us,
                    num_history_steps=self.num_history_steps,
                    num_future_steps=self.num_future_steps,
                    time_step=self.time_step,
                )
            except BadLocalZipError as e:
                # Should have been caught at init, but zips can go bad later
                # (NFS hiccup, etc). Surface a clean error pointing at the clip.
                raise RuntimeError(
                    f"Local zip unreadable for clip {clip_id} at ts={t0_us}: {e}. "
                    f"Re-run the dataset init to refresh the bad-clip list, "
                    f"or exclude this clip from the COC JSONL."
                ) from e
        else:
            sample_data = self._load_from_cache(clip_id, t0_us)
            # Cache pkl was built with include_extr_intr=True, so it carries
            # extr/intr/vehicle_dimensions. Local path doesn't (unless the
            # caller opted in). Strip them here so a mixed batch has a
            # consistent field set and the collate doesn't KeyError.
            if not self.include_extr_intr:
                for k in ("extr", "intr", "vehicle_dimensions"):
                    sample_data.pop(k, None)

        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.include_extr_intr:
            if not hasattr(self, "_avdi"):
                self._avdi = _LocalPAIFeatureReader(
                    local_dir=self.local_dir,
                    clip_index_metadata=self.clip_index_metadata,
                )
            sample_data["extr"] = self._avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample_data["intr"] = self._avdi.get_clip_feature(clip_id, "camera_intrinsics")

            vehicle_dimensions = self._avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            sample_data["ego_lwh"] = torch.tensor(
                [vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]
            )
            sample_data["ego_length_offset"] = torch.tensor(
                vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length
            )

        # Inject CoC reasoning text
        sample_data["cot"] = sample["cot"]

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
