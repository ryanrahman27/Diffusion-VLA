"""
Merge Ishan-Axibo/piperx_laundry_ee with Facebear/XVLA-Soft-Fold (one folder) into a
single LeRobot EE dataset with distinct language instructions per source.

Base dataset (copied as-is):
    Ishan-Axibo/piperx_laundry_ee  ->  task: "pick towel from pile, fold and stack"

Appended from HDF5 (EE 6D, cam_high mapped to cam_front):
    Facebear/XVLA-Soft-Fold/<xvla-folder>/  ->  task: "soft fold the cloth"

XVLA episodes are downloaded one at a time and deleted after conversion to limit
disk usage (~50 GB for the full folder if downloaded all at once).

Usage:
    uv run python examples/piper_real/ee/merge_laundry_softfold_ee.py \\
        --output-repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged

Resume after a partial run (laundry already copied, append remaining XVLA):
    uv run python examples/piper_real/ee/merge_laundry_softfold_ee.py \\
        --output-repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged \\
        --no-overwrite --skip-laundry-copy --resume

Then:
    uv run scripts/compute_norm_stats.py --config-name pi05_piperx_laundry_softfold_ee_merged
    uv run scripts/train.py pi05_piperx_laundry_softfold_ee_merged --exp-name=merged_v1 --overwrite
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import cv2
import h5py
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.video_utils import get_safe_default_codec
import numpy as np
import torch
import tqdm
import tyro

EEF_6D_KEY = "observations/eef_6d"
EEF_6D_DIM = 20
FPS = 30

OUTPUT_CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")
XVLA_CAMERA_MAP = {
    "cam_front": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig(
    image_writer_processes=2,
    image_writer_threads=4,
)


def _natural_sort_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _next_step_actions(pose: np.ndarray) -> np.ndarray:
    if pose.shape[0] == 0:
        return pose
    if pose.shape[0] == 1:
        return pose.copy()
    actions = np.empty_like(pose)
    actions[:-1] = pose[1:]
    actions[-1] = pose[-1]
    return actions


def _count_existing_episodes(dataset_root: Path) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return 0
    return sum(1 for line in episodes_path.read_text().splitlines() if line.strip())


def copy_laundry_base_dataset(
    *,
    source_repo_id: str,
    output_repo_id: str,
    overwrite: bool,
) -> Path:
    target = Path(LEROBOT_HOME / output_repo_id)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists at {target}. "
                "Pass --no-overwrite --skip-laundry-copy to resume."
            )
        print(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    print(f"Downloading {source_repo_id} -> {target}")
    snapshot_download(
        source_repo_id,
        repo_type="dataset",
        local_dir=target,
    )
    return target


def open_dataset_for_append(
    repo_id: str,
    dataset_root: Path,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    root = dataset_root.resolve()
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.repo_id = repo_id
    obj.root = root
    obj.meta = LeRobotDatasetMetadata(repo_id, root=root)
    obj.revision = None
    obj.tolerance_s = dataset_config.tolerance_s
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.episodes = None
    obj.episode_data_index = None
    obj.image_writer = None
    obj.video_backend = dataset_config.video_backend or get_safe_default_codec()

    if dataset_config.image_writer_processes or dataset_config.image_writer_threads:
        obj.start_image_writer(
            dataset_config.image_writer_processes,
            dataset_config.image_writer_threads,
        )

    obj.episode_buffer = obj.create_episode_buffer()
    obj.hf_dataset = obj.create_hf_dataset()
    print(
        f"Append mode: {obj.meta.total_episodes} episode(s), "
        f"{obj.meta.total_frames} frame(s) on disk."
    )
    return obj


def list_xvla_hdf5_paths(
    *,
    xvla_repo_id: str,
    xvla_folder: str,
) -> list[str]:
    prefix = f"{xvla_folder}/"
    paths = [
        path
        for path in list_repo_files(xvla_repo_id, repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".hdf5")
    ]
    paths.sort(key=lambda p: _natural_sort_key(Path(p)))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files found under {xvla_repo_id}/{xvla_folder}. "
            f"Check --xvla-folder={xvla_folder!r}."
        )
    print(f"Found {len(paths)} XVLA episodes in {xvla_folder}")
    return paths


def load_xvla_episode(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        if EEF_6D_KEY not in ep:
            raise KeyError(f"{ep_path}: missing {EEF_6D_KEY}")
        eef_6d = np.asarray(ep[EEF_6D_KEY][:], dtype=np.float32)
        if eef_6d.shape[-1] != EEF_6D_DIM:
            raise ValueError(f"{ep_path}: expected {EEF_6D_DIM}-D eef_6d, got {eef_6d.shape}")
        state = torch.from_numpy(eef_6d)
        action = torch.from_numpy(_next_step_actions(eef_6d))

        imgs_per_cam: dict[str, np.ndarray] = {}
        for out_cam, src_cam in XVLA_CAMERA_MAP.items():
            key = f"observations/images/{src_cam}"
            if key not in ep:
                raise KeyError(f"{ep_path}: missing camera dataset {key}")
            ds = ep[key]
            decoded = []
            for jpeg_bytes in ds:
                arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"{ep_path}: failed to decode a JPEG in {src_cam}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                decoded.append(img_rgb)
            imgs_per_cam[out_cam] = np.stack(decoded, axis=0)

    return imgs_per_cam, state, action


def append_xvla_episode(
    dataset: LeRobotDataset,
    ep_path: Path,
    *,
    task: str,
) -> None:
    imgs_per_cam, state, action = load_xvla_episode(ep_path)
    num_frames = state.shape[0]

    for cam in OUTPUT_CAMERAS:
        if imgs_per_cam[cam].shape[0] != num_frames:
            raise ValueError(
                f"{ep_path}: camera {cam} has {imgs_per_cam[cam].shape[0]} "
                f"frames but state has {num_frames}"
            )
    if action.shape[0] != num_frames:
        raise ValueError(
            f"{ep_path}: action has {action.shape[0]} frames but state has {num_frames}"
        )

    for i in range(num_frames):
        frame = {
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        }
        for cam in OUTPUT_CAMERAS:
            frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]
        dataset.add_frame(frame)

    dataset.save_episode()


def append_xvla_episodes_streaming(
    dataset: LeRobotDataset,
    *,
    xvla_repo_id: str,
    xvla_paths: list[str],
    task: str,
    xvla_cache_dir: Path,
    delete_after_convert: bool,
    skip_count: int = 0,
) -> None:
    xvla_cache_dir = xvla_cache_dir.expanduser().resolve()
    xvla_cache_dir.mkdir(parents=True, exist_ok=True)

    if skip_count:
        print(f"Skipping first {skip_count} XVLA episode(s) already in the dataset.")
        xvla_paths = xvla_paths[skip_count:]

    for repo_path in tqdm.tqdm(xvla_paths, desc="XVLA episodes"):
        local_path = Path(
            hf_hub_download(
                xvla_repo_id,
                repo_path,
                repo_type="dataset",
                local_dir=xvla_cache_dir,
            )
        )
        append_xvla_episode(dataset, local_path, task=task)
        if delete_after_convert:
            local_path.unlink(missing_ok=True)


def print_task_summary(dataset_root: Path) -> None:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    print("\nMerged dataset tasks:")
    for line in tasks_path.read_text().splitlines():
        if line.strip():
            print(f"  {line.strip()}")
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        print(
            f"Total: {info.get('total_episodes')} episodes, "
            f"{info.get('total_frames')} frames"
        )


def main(
    *,
    output_repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    laundry_repo_id: str = "Ishan-Axibo/piperx_laundry_ee",
    xvla_repo_id: str = "Facebear/XVLA-Soft-Fold",
    xvla_folder: str = "0714_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve",
    laundry_task: str = "pick towel from pile, fold and stack",
    xvla_task: str = "soft fold the cloth",
    xvla_cache_dir: Path = Path("~/.cache/openpi/xvla_softfold_hdf5"),
    overwrite: bool = True,
    skip_laundry_copy: bool = False,
    resume: bool = False,
    laundry_episode_count: int = 148,
    delete_xvla_after_convert: bool = True,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    target = Path(LEROBOT_HOME / output_repo_id)

    if skip_laundry_copy:
        if not target.exists() or _count_existing_episodes(target) == 0:
            raise FileNotFoundError(
                f"--skip-laundry-copy set but no dataset found at {target}. "
                "Run without --skip-laundry-copy first."
            )
        dataset_root = target
        print(f"Reusing existing laundry base at {dataset_root}")
    else:
        dataset_root = copy_laundry_base_dataset(
            source_repo_id=laundry_repo_id,
            output_repo_id=output_repo_id,
            overwrite=overwrite,
        )

    xvla_paths = list_xvla_hdf5_paths(
        xvla_repo_id=xvla_repo_id,
        xvla_folder=xvla_folder,
    )

    xvla_skip = 0
    if resume:
        existing = _count_existing_episodes(dataset_root)
        xvla_skip = max(0, existing - laundry_episode_count)
        if xvla_skip >= len(xvla_paths):
            print(
                f"All {len(xvla_paths)} XVLA episodes already present "
                f"({existing} total episodes). Nothing to do."
            )
            print_task_summary(dataset_root)
            return
        print(
            f"Resume: {existing} episodes on disk "
            f"({laundry_episode_count} laundry + {xvla_skip} XVLA)."
        )

    dataset = open_dataset_for_append(
        output_repo_id,
        dataset_root,
        dataset_config=dataset_config,
    )
    append_xvla_episodes_streaming(
        dataset,
        xvla_repo_id=xvla_repo_id,
        xvla_paths=xvla_paths,
        task=xvla_task,
        xvla_cache_dir=xvla_cache_dir,
        delete_after_convert=delete_xvla_after_convert,
        skip_count=xvla_skip,
    )

    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    print_task_summary(dataset_root)
    print(f"\nMerged dataset written to {dataset_root.resolve()}")
    print(f"  Laundry source: {laundry_repo_id} (task={laundry_task!r})")
    print(f"  XVLA source: {xvla_repo_id}/{xvla_folder} (task={xvla_task!r})")

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print("Skipped Hub push (pass --push-to-hub to upload).")


if __name__ == "__main__":
    tyro.cli(main)
