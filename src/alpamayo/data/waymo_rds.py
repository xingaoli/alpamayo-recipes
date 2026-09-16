# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Waymo rds_hq dataset emitting PAI-contract samples for SFT.

Ported from ``alpamayo1.5/waymo_eval/rds_dataset.py`` (the inference-side
waymo eval loader). ``RDSWaymoClip.build_sample`` emits a dict in the exact
contract of ``load_physical_aiavdataset_local`` (the PAI loader used by the
alpamayo SFT pipeline), so the processor / collate / model side needs no
changes. ``RDSWaymoDataset`` wraps the clips in a flat ``(clip_id, t0)`` index
with the ``PAIMultiSegmentDataset`` constructor surface
(``vla_preprocess_args`` / ``model_config``) so it drops into the Hydra
configs unchanged.

Layout (per split ``train`` / ``val``, per segment ``<seg>``):

  pinhole_front/<seg>.mp4 (etc.)    1920x1280 @10Hz camera videos (198 frames)
  vehicle_pose/<seg>.tar            {seg}.{idx:06d}.vehicle_pose.npy    4x4 world pose
  timestamp/<seg>.tar               {seg}.{idx:06d}.timestamp_micros.txt  int (us)

The stream tars hold data on a 30 Hz frame grid (idx 0..593); the real 10 Hz
keyframes sit at ``idx % 3 == 0`` and only those carry timestamps. The mp4s
contain exactly the 198 keyframes, so mp4 frame ``i`` is keyframe ``i``. The
egomotion is built from the 198 keyframe ``vehicle_pose`` entries + their true
timestamps and interpolated onto the 0.1 s query grid (mirroring the PAI
``EgomotionState`` interpolator, which interpolates 100 Hz egomotion).

Camera mapping onto the trained 4-cam PAI layout
(cross_left=0, front_wide=1, cross_right=2, front_tele=6):

  pinhole_front_left -> 0, pinhole_front -> 1, pinhole_front_right -> 2,
  and the front_tele slot is filled with pinhole_front (waymo has no narrow
  FOV camera) — the same trick as the original waymo_eval script.

Ego reference frame is the *vehicle* pose (rds_hq ships real vehicle poses,
unlike the old extracted data where the front camera doubled as ego).

Eval-only streams (pose / pinhole_intrinsic / all_object_info / 3d_lanelines
...) are not read here; they stay in waymo_eval.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")

# (rds camera stream name, model camera index) — must stay sorted by model idx.
CAMERA_SLOTS: tuple[tuple[str, int], ...] = (
    ("pinhole_front_left", 0),   # PAI cross_left slot
    ("pinhole_front", 1),        # PAI front_wide slot
    ("pinhole_front_right", 2),  # PAI cross_right slot
    ("pinhole_front", 6),        # PAI front_tele slot (filled from front)
)
FRONT_CAM = "pinhole_front"


def _read_tar_members(tar_path: Path, suffix_filter: str | None = None
                      ) -> dict[str, bytes]:
    """Read all members of a small metadata tar into memory ({name: bytes})."""
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if suffix_filter is not None and not m.name.endswith(suffix_filter):
                continue
            f = tf.extractfile(m)
            if f is not None:
                out[m.name] = f.read()
    return out


def _member_idx(name: str, seg: str) -> int:
    """``<seg>.000123.suffix`` -> 123."""
    stem = name.split("/")[-1]
    return int(stem[len(seg) + 1:].split(".")[0])


class RDSWaymoClip:
    """Lazy per-segment reader for one rds_hq clip."""

    def __init__(self, root: Path, split: str, clip_id: str, preload: bool = False):
        self.root = Path(root)
        self.split = split
        self.clip_id = clip_id
        self.preload = preload

        # --- timestamps (canonical 10Hz keyframe grid) ---
        ts_raw = _read_tar_members(
            self.root / split / "timestamp" / f"{clip_id}.tar",
            suffix_filter=".timestamp_micros.txt")
        if not ts_raw:
            raise RuntimeError(f"empty timestamp tar for {clip_id}")
        pairs = sorted((_member_idx(n, clip_id), int(content.decode().strip()))
                       for n, content in ts_raw.items())
        self.keyframe_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
        self.ts_us = np.asarray([p[1] for p in pairs], dtype=np.int64)
        if not np.all(np.diff(self.ts_us) > 0):
            raise RuntimeError(f"non-monotonic timestamps in {clip_id}")
        self.n_keyframes = len(self.ts_us)

        # --- vehicle poses at keyframes (4x4, world frame) ---
        vp_raw = _read_tar_members(
            self.root / split / "vehicle_pose" / f"{clip_id}.tar",
            suffix_filter=".vehicle_pose.npy")
        if not vp_raw:
            raise RuntimeError(f"empty vehicle_pose tar for {clip_id}")
        poses = np.zeros((self.n_keyframes, 4, 4), dtype=np.float64)
        poses[:, 3, 3] = 1.0
        n_filled = 0
        for n, content in vp_raw.items():
            idx = _member_idx(n, clip_id)
            i = np.searchsorted(self.keyframe_idx, idx)
            if i < self.n_keyframes and self.keyframe_idx[i] == idx:
                poses[i] = np.load(io.BytesIO(content))
                n_filled += 1
        if n_filled != self.n_keyframes:
            raise RuntimeError(
                f"{clip_id}: vehicle_pose tar covers {n_filled}/"
                f"{self.n_keyframes} keyframes")
        self.vehicle_poses = poses

        t_sec = self.ts_us.astype(np.float64) * 1e-6
        self._slerp = Slerp(t_sec, Rotation.from_matrix(poses[:, :3, :3]))

        # --- videos ---
        self._videos: dict[str, np.ndarray | cv2.VideoCapture] = {}
        if preload:
            for cam in {c for c, _ in CAMERA_SLOTS}:
                self._videos[cam] = self._decode_video(cam)

    # ------------------------------------------------------------------ videos
    def _video_path(self, cam: str) -> Path:
        return self.root / self.split / cam / f"{self.clip_id}.mp4"

    def _decode_video(self, cam: str) -> np.ndarray:
        cap = cv2.VideoCapture(str(self._video_path(cam)))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self._video_path(cam)}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded: {self._video_path(cam)}")
        return np.stack(frames)  # [T, H, W, 3] RGB

    def decode_frames(self, cam: str, frame_idx: np.ndarray) -> np.ndarray:
        """Decode keyframe indices -> [T, H, W, 3] RGB."""
        v = self._videos.get(cam)
        if isinstance(v, np.ndarray):
            if frame_idx.max() >= v.shape[0] or frame_idx.min() < 0:
                raise IndexError(
                    f"{self.clip_id}/{cam}: frame {frame_idx} outside "
                    f"[0, {v.shape[0] - 1}]")
            return v[frame_idx]
        cap = (v if isinstance(v, cv2.VideoCapture)
               else cv2.VideoCapture(str(self._video_path(cam))))
        self._videos[cam] = cap
        out = []
        for i in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{self.clip_id}/{cam}: seek to {i} failed")
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    # ------------------------------------------------------------------ poses
    def pose_at(self, ts_us) -> np.ndarray:
        """Interpolated vehicle pose [N, 4, 4] at absolute timestamps (us)."""
        t = np.atleast_1d(np.asarray(ts_us, dtype=np.float64)) * 1e-6
        t0, t1 = self.ts_us[0] * 1e-6, self.ts_us[-1] * 1e-6
        if t.min() < t0 or t.max() > t1:
            raise ValueError(
                f"{self.clip_id}: query t=[{t.min():.3f},{t.max():.3f}]s outside "
                f"clip span [{t0:.3f},{t1:.3f}]s")
        xyz = np.stack(
            [np.interp(t, self.ts_us * 1e-6, self.vehicle_poses[:, k, 3])
             for k in range(3)], axis=1)
        rot = self._slerp(t).as_matrix()
        out = np.zeros((len(t), 4, 4), dtype=np.float64)
        out[:, :3, :3] = rot
        out[:, :3, 3] = xyz
        out[:, 3, 3] = 1.0
        return out

    def keyframe_index_at(self, ts_us: int) -> int:
        """Nearest keyframe index for an absolute timestamp."""
        return int(np.argmin(np.abs(self.ts_us - ts_us)))

    # ------------------------------------------------------------------ samples
    def segment_t0s(self, num_segments: int = 20, segment_start_us: int = 2_000_000,
                    segment_step_us: int = 1_000_000, num_history_steps: int = 16,
                    num_future_steps: int = 64, time_step: float = 0.1
                    ) -> list[int]:
        """PAIMultiSegmentDataset schedule, filtered by the real clip span."""
        hist_us = int((num_history_steps - 1) * time_step * 1_000_000)
        fut_us = int(num_future_steps * time_step * 1_000_000)
        t0s = []
        for k in range(num_segments):
            t0 = int(self.ts_us[0]) + segment_start_us + k * segment_step_us
            if t0 - hist_us >= int(self.ts_us[0]) and t0 + fut_us <= int(self.ts_us[-1]):
                t0s.append(t0)
        return t0s

    def build_sample(self, t0_us: int, num_history_steps: int = 16,
                     num_future_steps: int = 64, time_step: float = 0.1,
                     num_frames: int = 4) -> dict:
        """One PAI-contract sample at ``t0_us`` (absolute micros).

        Key-for-key and shape-for-shape identical to
        ``load_physical_aiavdataset_local`` (ego tensors pre-squeeze:
        ``[1, 1, T, 3]`` / ``[1, 1, T, 3, 3]``).
        """
        dt_us = int(time_step * 1_000_000)

        hist_ts = t0_us + np.arange(
            -(num_history_steps - 1) * dt_us, dt_us / 2, dt_us).astype(np.int64)
        fut_ts = t0_us + np.arange(
            dt_us, (num_future_steps + 0.5) * dt_us, dt_us).astype(np.int64)

        hist = self.pose_at(hist_ts)
        fut = self.pose_at(fut_ts)
        t0_pose = self.pose_at(t0_us)[0]
        t0_xyz = t0_pose[:3, 3]
        t0_rot = t0_pose[:3, :3]
        t0_rot_inv = t0_rot.T

        def to_local(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xyz = (t0_rot_inv @ (poses[:, :3, 3] - t0_xyz).T).T
            rot = np.einsum("ij,njk->nik", t0_rot_inv, poses[:, :3, :3])
            return xyz, rot

        hist_xyz_l, hist_rot_l = to_local(hist)
        fut_xyz_l, fut_rot_l = to_local(fut)

        # --- images: num_frames timestamps ending at t0 ---
        img_ts = np.asarray(
            [t0_us - (num_frames - 1 - i) * dt_us for i in range(num_frames)],
            dtype=np.int64)
        img_idx = np.asarray([self.keyframe_index_at(t) for t in img_ts])

        frames_list, cam_idx_list, ts_list = [], [], []
        for cam, model_idx in CAMERA_SLOTS:
            frames = self.decode_frames(cam, img_idx)  # [T,H,W,3]
            ft = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            frames_list.append(ft)
            cam_idx_list.append(model_idx)
            ts_list.append(torch.from_numpy(img_ts))
        image_frames = torch.stack(frames_list, dim=0)   # [C, T, 3, H, W]
        camera_indices = torch.tensor(cam_idx_list, dtype=torch.int64)
        all_ts = torch.stack(ts_list, dim=0)             # [C, T]
        rel_ts = (all_ts - all_ts.min()).float() * 1e-6

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": torch.from_numpy(hist_xyz_l).float()[None, None],
            "ego_history_rot": torch.from_numpy(hist_rot_l).float()[None, None],
            "ego_future_xyz": torch.from_numpy(fut_xyz_l).float()[None, None],
            "ego_future_rot": torch.from_numpy(fut_rot_l).float()[None, None],
            "relative_timestamps": rel_ts,
            "absolute_timestamps": all_ts,
            "t0_us": int(t0_us),
            "clip_id": self.clip_id,
        }


class RDSWaymoDataset(Dataset):
    """Flat ``(clip_id, t0)`` index over one split — PAIMultiSegmentDataset style.

    Iterating sequentially reuses one decoded-video cache; random access works
    too but re-decodes on clip switch. Keep ``preload=False`` for shuffled
    training: a preloaded clip holds ~4.4 GB of decoded frames per camera set.
    """

    def __init__(
        self,
        local_dir: str | None = None,
        split: str = "train",
        clip_ids: list[str] | None = None,
        num_segments: int = 20,
        segment_start_us: int = 2_000_000,
        segment_step_us: int = 1_000_000,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        num_frames: int = 4,
        preload: bool = False,
        max_clips: int | None = None,
        clip_stride: int = 1,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
    ):
        """Initialize the waymo multi-segment dataset.

        Args:
            local_dir: Root of the waymo rds_hq dataset (contains ``train/``
                and ``val/``). Falls back to the ``ALPAMAYO_WAYMO_DIR``
                environment variable (see the recipe ``.env``).
            split: Subdirectory to read (``train`` or ``val``).
            clip_ids: Optional explicit clip list (mp4 stems). Default: all
                clips in the split.
            num_segments: Number of segments (t0 samples) to draw per clip.
            segment_start_us: First t0 (microseconds) after clip start.
            segment_step_us: Spacing (microseconds) between consecutive t0s.
            num_history_steps: History length of each trajectory sample.
            num_future_steps: Future horizon of each trajectory sample.
            time_step: Seconds per step between trajectory samples.
            num_frames: Number of camera frames per sample (ending at t0).
            preload: Decode whole clip videos at first touch (~4.4 GB/clip).
                Only sensible for sequential iteration (e.g. eval).
            max_clips: Cap the clip list (after stride) — smoke tests.
            clip_stride: Take every Nth clip — smoke tests / subsampling.
            model_config: Optional config (dict is converted with OmegaConf)
                passed as ``model_config`` into ``hydra.utils.instantiate`` when
                building the preprocessor.
            vla_preprocess_args: If set, Hydra config dict to instantiate a
                preprocessor; its result is stored as ``tokenized_data`` on
                each sample.
        """
        if local_dir is None:
            local_dir = os.environ.get("ALPAMAYO_WAYMO_DIR", "data/waymo_rds_hq")
        assert local_dir and os.path.exists(local_dir), (
            f"Waymo rds_hq root not found: {local_dir}. "
            f"Set the ALPAMAYO_WAYMO_DIR env var or pass local_dir."
        )

        self.root = Path(local_dir)
        self.split = split
        self.num_segments = num_segments
        self.segment_start_us = segment_start_us
        self.segment_step_us = segment_step_us
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.num_frames = num_frames
        self.preload = preload

        clips = sorted(p.stem for p in (self.root / split / FRONT_CAM).glob("*.mp4"))
        if clip_ids is not None:
            want = set(clip_ids)
            clips = [c for c in clips if c in want]
        if clip_stride > 1:
            clips = clips[::clip_stride]
        if max_clips is not None:
            clips = clips[:max_clips]
        if not clips:
            raise RuntimeError(f"no clips under {self.root / split / FRONT_CAM}")

        self.clip_ids = clips
        self._clip_cache: dict[str, RDSWaymoClip] = {}
        self.index: list[tuple[str, int]] = []
        skipped: list[str] = []
        for cid in clips:
            try:
                clip = self._get_clip(cid)
                t0s = clip.segment_t0s(
                    num_segments, segment_start_us, segment_step_us,
                    num_history_steps, num_future_steps, time_step)
            except Exception as exc:
                skipped.append(f"{cid}: {exc}")
                continue
            self.index.extend((cid, t0) for t0 in t0s)
        if skipped:
            logger.warning(
                "Skipped %d/%d waymo clips (unreadable metadata or short span): %s%s",
                len(skipped), len(clips), "; ".join(skipped[:3]),
                " ..." if len(skipped) > 3 else "",
            )
        if not self.index:
            raise RuntimeError(
                f"no (clip, t0) samples indexed under {self.root / split}")

        # ----- optional VLA preprocessor (same wiring as PAIMultiSegmentDataset) -----
        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(
                vla_preprocess_args, model_config=model_config
            )

        logger.info(
            "RDSWaymoDataset: split=%s, %d clips, %d samples "
            "(num_segments=%d, start=%dus, step=%dus)",
            split, len(clips), len(self.index), num_segments,
            segment_start_us, segment_step_us,
        )

    def _get_clip(self, clip_id: str) -> RDSWaymoClip:
        if clip_id not in self._clip_cache:
            if len(self._clip_cache) >= 2:  # keep at most 2 decoded clips alive
                self._clip_cache.clear()
            self._clip_cache[clip_id] = RDSWaymoClip(
                self.root, self.split, clip_id, preload=self.preload)
        return self._clip_cache[clip_id]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id, t0_us = self.index[idx]
        clip = self._get_clip(clip_id)
        sample_data = clip.build_sample(
            t0_us,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
            num_frames=self.num_frames,
        )

        # squeeze ego motion shape (matches PAIMultiSegmentDataset, which
        # squeezes the [1, 1, T, ...] loader output once)
        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
