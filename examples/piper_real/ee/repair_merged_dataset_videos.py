"""Repair corrupt LeRobot MP4s in the merged laundry+XVLA EE dataset.

Scans all episode videos with ffprobe and re-encodes any that fail from the
original XVLA HDF5 source (episodes >= laundry_episode_count).

Usage:
    uv run python examples/piper_real/ee/repair_merged_dataset_videos.py

    uv run python examples/piper_real/ee/repair_merged_dataset_videos.py \\
        --episode 153 --camera cam_left_wrist
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
from huggingface_hub import hf_hub_download
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.video_utils import encode_video_frames
import numpy as np
from PIL import Image
import tyro

XVLA_FOLDER = "0714_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve"
XVLA_REPO = "Facebear/XVLA-Soft-Fold"
XVLA_CAMERA_MAP = {
    "cam_front": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
FPS = 30


def _video_ok(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", str(path)],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _find_bad_videos(dataset_root: Path) -> list[tuple[int, str, Path]]:
    bad: list[tuple[int, str, Path]] = []
    for cam in XVLA_CAMERA_MAP:
        cam_dir = dataset_root / "videos" / "chunk-000" / f"observation.images.{cam}"
        if not cam_dir.exists():
            continue
        for path in sorted(cam_dir.glob("episode_*.mp4")):
            if not _video_ok(path):
                ep_idx = int(path.stem.split("_")[-1])
                bad.append((ep_idx, cam, path))
    return bad


def _decode_camera_frames(hdf5_path: Path, src_cam: str) -> list[Image.Image]:
    key = f"observations/images/{src_cam}"
    with h5py.File(hdf5_path, "r") as ep:
        if key not in ep:
            raise KeyError(f"{hdf5_path}: missing {key}")
        frames: list[Image.Image] = []
        for jpeg_bytes in ep[key]:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"{hdf5_path}: failed to decode JPEG in {src_cam}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))
    return frames


def _xvla_hdf5_path(
    *,
    episode_index: int,
    laundry_episode_count: int,
    xvla_repo: str,
    xvla_folder: str,
    cache_dir: Path,
) -> Path:
    if episode_index < laundry_episode_count:
        raise ValueError(
            f"Episode {episode_index} is laundry (< {laundry_episode_count}). "
            "Re-convert from the laundry source HDF5 manually."
        )
    xvla_ep = episode_index - laundry_episode_count
    repo_path = f"{xvla_folder}/episode_{xvla_ep}.hdf5"
    return Path(
        hf_hub_download(
            xvla_repo,
            repo_path,
            repo_type="dataset",
            local_dir=cache_dir,
        )
    )


def repair_video(
    *,
    dataset_root: Path,
    episode_index: int,
    camera: str,
    laundry_episode_count: int = 148,
    xvla_repo: str = XVLA_REPO,
    xvla_folder: str = XVLA_FOLDER,
    cache_dir: Path = Path("~/.cache/openpi/xvla_repair_hdf5"),
) -> None:
    if camera not in XVLA_CAMERA_MAP:
        raise ValueError(f"Unknown camera {camera!r}. Choose from {list(XVLA_CAMERA_MAP)}")

    video_path = (
        dataset_root
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera}"
        / f"episode_{episode_index:06d}.mp4"
    )
    if not video_path.parent.exists():
        raise FileNotFoundError(f"Video directory missing: {video_path.parent}")

    cache_dir = cache_dir.expanduser().resolve()
    hdf5_path = _xvla_hdf5_path(
        episode_index=episode_index,
        laundry_episode_count=laundry_episode_count,
        xvla_repo=xvla_repo,
        xvla_folder=xvla_folder,
        cache_dir=cache_dir,
    )
    src_cam = XVLA_CAMERA_MAP[camera]
    frames = _decode_camera_frames(hdf5_path, src_cam)

    backup = video_path.with_suffix(".mp4.bak")
    if video_path.exists():
        shutil.move(video_path, backup)

    with tempfile.TemporaryDirectory(prefix="repair_frames_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            frame.save(tmp_dir / f"frame_{i:06d}.png")
        encode_video_frames(
            tmp_dir,
            video_path,
            fps=FPS,
            vcodec="libsvtav1",
            overwrite=True,
        )

    if not _video_ok(video_path):
        if backup.exists():
            shutil.move(backup, video_path)
        raise RuntimeError(f"Re-encoded video still invalid: {video_path}")

    if backup.exists():
        backup.unlink()
    print(f"Repaired {video_path} ({len(frames)} frames from {hdf5_path.name})")


def main(
    *,
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    laundry_episode_count: int = 148,
    episode: int | None = None,
    camera: str | None = None,
    scan_only: bool = False,
) -> None:
    dataset_root = Path(LEROBOT_HOME / repo_id)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    if episode is not None:
        if camera is None:
            raise ValueError("Pass --camera when using --episode")
        repair_video(
            dataset_root=dataset_root,
            episode_index=episode,
            camera=camera,
            laundry_episode_count=laundry_episode_count,
        )
        return

    bad = _find_bad_videos(dataset_root)
    if not bad:
        print(f"No corrupt videos found in {dataset_root}")
        return

    print(f"Found {len(bad)} corrupt video(s):")
    for ep_idx, cam, path in bad:
        print(f"  episode {ep_idx} {cam}: {path}")

    if scan_only:
        return

    for ep_idx, cam, _path in bad:
        repair_video(
            dataset_root=dataset_root,
            episode_index=ep_idx,
            camera=cam,
            laundry_episode_count=laundry_episode_count,
        )


if __name__ == "__main__":
    tyro.cli(main)
