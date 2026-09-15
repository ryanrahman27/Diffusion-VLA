"""
Convert PiperX HIL rollout HDF5 episodes (``piperx_hil_v1`` schema, written by
``hil_episode.write_hil_hdf5`` from the OpenPI inference stack) into the
LeRobot v2.0 dataset format that openpi consumes.

This is the rollout counterpart to ``convert_piperx_data_to_lerobot.py`` (teleop
``episode_recorder.py`` output). Do **not** use the teleop converter on HIL
rollout files — the HDF5 layout differs.

What this script does:
    * Reads action/achieved as observation.state  (14-dim float32, follower proprio)
    * Reads action/commanded as action           (14-dim float32, policy/human cmd)
    * Decodes the three JPEG camera streams (cam_front, cam_left_wrist,
      cam_right_wrist) from observation/image_<cam>/jpeg into uint8 HWC arrays
    * Reads metadata/task_description as the per-frame "task" string
    * Skips aborted episodes and optionally skips failure/timeout rollouts

State / action layout (14-dim, both):
    [L_arm_6 (rad), L_grip (m), R_arm_6 (rad), R_grip (m)]

Usage (single episode smoke test):
    uv run python examples/piper_real/convert_piperx_rollouts_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/laundry_rollouts_success_only \\
        --repo-id Ishan-Axibo/piperx_flatten_success_rollouts \\
        --task "pick towel from pile, fold and stack" \\
        --episodes 0

Usage (full conversion):
    uv run python examples/piper_real/convert_piperx_rollouts_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/laundry_rollouts_success_only \\
        --repo-id Ishan-Axibo/piperx_flatten_success_rollouts \\
        --task "pick towel from pile, fold and stack"

Usage (resume after interrupt — same raw-dir and repo-id as before):
    uv run python examples/piper_real/convert_piperx_rollouts_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/laundry_rollouts_success_only \\
        --repo-id Ishan-Axibo/piperx_flatten_success_rollouts \\
        --task "pick towel from pile, fold and stack" \\
        --resume
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

HIL_SCHEMA = "piperx_hil_v1"

CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")

MOTORS = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
]

STATE_DIM = len(MOTORS)
FPS = 30
CONVERSION_SCRIPT = "convert_piperx_rollouts_to_lerobot"
CONVERSION_STATE_VERSION = 1
CONVERSION_STATE_REL = Path("meta") / "conversion_state.json"


@dataclasses.dataclass
class ConversionProgress:
    """Checkpoint for HDF5→LeRobot bulk conversion (enables --resume)."""

    script: str
    raw_dir: str
    repo_id: str
    max_episodes: int | None
    processed_sources: list[str] = dataclasses.field(default_factory=list)
    skip_counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def processed_set(self) -> set[str]:
        return set(self.processed_sources)

    def to_json_dict(self) -> dict:
        return {
            "version": CONVERSION_STATE_VERSION,
            "script": self.script,
            "raw_dir": self.raw_dir,
            "repo_id": self.repo_id,
            "max_episodes": self.max_episodes,
            "processed_sources": self.processed_sources,
            "skip_counts": self.skip_counts,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> ConversionProgress:
        version = int(data.get("version", 0))
        if version != CONVERSION_STATE_VERSION:
            raise ValueError(
                f"Unsupported {CONVERSION_STATE_REL} version {version} "
                f"(expected {CONVERSION_STATE_VERSION})"
            )
        return cls(
            script=str(data["script"]),
            raw_dir=str(data["raw_dir"]),
            repo_id=str(data["repo_id"]),
            max_episodes=data.get("max_episodes"),
            processed_sources=list(data.get("processed_sources", [])),
            skip_counts={str(k): int(v) for k, v in dict(data.get("skip_counts", {})).items()},
        )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _attr_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray | None:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _decode_jpeg_list_hold_last(
    jpegs: list[bytes],
    *,
    fill_shape: tuple[int, int, int] = (480, 640, 3),
) -> np.ndarray:
    """Decode frames; repeat last good frame on failure; zero-fill if none decoded."""
    filled: list[np.ndarray] = []
    last_good: np.ndarray | None = None
    zeros = np.zeros(fill_shape, dtype=np.uint8)
    for raw in jpegs:
        img = _decode_jpeg_bytes(raw)
        if img is None:
            img = last_good
        if img is None:
            img = zeros
        else:
            last_good = img
        filled.append(img)
    return np.stack(filled, axis=0)


def conversion_state_path(dataset_root: Path) -> Path:
    return dataset_root / CONVERSION_STATE_REL


def load_conversion_progress(dataset_root: Path) -> ConversionProgress | None:
    path = conversion_state_path(dataset_root)
    if not path.is_file():
        return None
    return ConversionProgress.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def save_conversion_progress(dataset_root: Path, progress: ConversionProgress) -> None:
    path = conversion_state_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(progress.to_json_dict(), indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(payload)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def filter_pending_hdf5(
    hdf5_files: list[Path],
    progress: ConversionProgress | None,
) -> list[Path]:
    if progress is None:
        return hdf5_files
    done = progress.processed_set()
    return [p for p in hdf5_files if p.name not in done]


def _episode_index_from_stem(stem: str) -> int | None:
    if not stem.startswith("episode_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def cleanup_stale_episode_artifacts(dataset_root: Path, *, completed_episodes: int) -> list[int]:
    """Remove parquet/videos for episodes that did not finish save_episode()."""
    removed: list[int] = []
    root = dataset_root.resolve()

    for pq in (root / "data").rglob("episode_*.parquet"):
        ep_idx = _episode_index_from_stem(pq.stem)
        if ep_idx is not None and ep_idx >= completed_episodes:
            pq.unlink(missing_ok=True)
            if ep_idx not in removed:
                removed.append(ep_idx)

    videos_root = root / "videos"
    if videos_root.is_dir():
        for mp4 in videos_root.rglob("episode_*.mp4"):
            ep_idx = _episode_index_from_stem(mp4.stem)
            if ep_idx is not None and ep_idx >= completed_episodes:
                mp4.unlink(missing_ok=True)
                if ep_idx not in removed:
                    removed.append(ep_idx)

    return sorted(removed)


def _assert_resume_compatible(
    progress: ConversionProgress,
    *,
    raw_dir: Path,
    repo_id: str,
    max_episodes: int | None,
) -> None:
    if progress.script != CONVERSION_SCRIPT:
        raise ValueError(
            f"Cannot resume: checkpoint is for {progress.script!r}, not {CONVERSION_SCRIPT!r}."
        )
    if progress.repo_id != repo_id:
        raise ValueError(
            f"Cannot resume: checkpoint repo_id {progress.repo_id!r} != {repo_id!r}."
        )
    if Path(progress.raw_dir).resolve() != raw_dir.resolve():
        raise ValueError(
            f"Cannot resume: checkpoint raw_dir {progress.raw_dir!r} "
            f"!= {raw_dir.resolve()!r}."
        )
    if progress.max_episodes != max_episodes:
        raise ValueError(
            f"Cannot resume: checkpoint max_episodes={progress.max_episodes!r} "
            f"!= current {max_episodes!r}. Use the same --max-episodes as the original run."
        )


def open_dataset_for_append(
    repo_id: str,
    dataset_root: Path,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Reopen a dataset for append without loading all parquet into RAM."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.video_utils import get_safe_default_codec

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


def ensure_conversion_progress(
    dataset_root: Path,
    *,
    raw_dir: Path,
    repo_id: str,
    max_episodes: int | None,
    existing: ConversionProgress | None,
) -> ConversionProgress:
    if existing is not None:
        return existing
    progress = ConversionProgress(
        script=CONVERSION_SCRIPT,
        raw_dir=str(raw_dir.resolve()),
        repo_id=repo_id,
        max_episodes=max_episodes,
    )
    save_conversion_progress(dataset_root, progress)
    return progress


def mark_source_processed(
    dataset_root: Path,
    progress: ConversionProgress,
    source_name: str,
    *,
    skip_reason: str | None = None,
) -> None:
    if source_name in progress.processed_set():
        return
    progress.processed_sources.append(source_name)
    if skip_reason:
        progress.skip_counts[skip_reason] = int(progress.skip_counts.get(skip_reason, 0)) + 1
    save_conversion_progress(dataset_root, progress)


def open_conversion_dataset(
    repo_id: str,
    *,
    raw_dir: Path,
    max_episodes: int | None,
    resume: bool,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> tuple[LeRobotDataset, Path, ConversionProgress | None, bool]:
    """Create a new dataset or reopen an in-progress one for --resume."""
    target = Path(LEROBOT_HOME / repo_id)
    raw_dir = raw_dir.resolve()

    if resume:
        if not target.is_dir():
            print(f"Resume: no dataset at {target}; starting a new conversion.")
            dataset = create_empty_dataset(repo_id, dataset_config=dataset_config)
            return dataset, target, None, False

        progress = load_conversion_progress(target)
        if progress is None:
            raise RuntimeError(
                f"Resume requested but {conversion_state_path(target)} is missing.\n"
                "This dataset was not produced by a resumable conversion run. "
                "Remove the output directory and convert again without --resume."
            )
        _assert_resume_compatible(
            progress,
            raw_dir=raw_dir,
            repo_id=repo_id,
            max_episodes=max_episodes,
        )
        print(
            f"Resuming conversion at {target}: "
            f"{len(progress.processed_sources)} source file(s) already processed."
        )
        info_path = target / "meta" / "info.json"
        completed_on_disk = (
            int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
            if info_path.is_file()
            else len(progress.processed_sources)
        )
        stale = cleanup_stale_episode_artifacts(target, completed_episodes=completed_on_disk)
        if stale:
            print(
                f"Removed stale artifacts from crashed episode(s): "
                f"{', '.join(f'{i:06d}' for i in stale)}"
            )
        dataset = open_dataset_for_append(repo_id, target, dataset_config=dataset_config)
        return dataset, target, progress, True

    if target.exists():
        print(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    dataset = create_empty_dataset(repo_id, dataset_config=dataset_config)
    return dataset, target, None, False


def classify_rollout_episode(
    end_reason: str,
    terminal: bool,
    truncated: bool,
) -> bool | None:
    """Return episode success, or None to skip the episode entirely."""
    if end_reason == "aborted":
        return None
    if end_reason == "success" and terminal:
        return True
    if end_reason == "failure" and terminal:
        return False
    if end_reason == "timeout" and truncated:
        return False
    return None


def create_empty_dataset(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [MOTORS],
        },
    }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if dataset_config.use_videos else "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="piperx_bimanual",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_episode(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None, bool | None, str]:
    """Read one PiperX HIL rollout HDF5 episode.

    Returns:
        (images_per_cam, state, action, instruction, is_success_or_none, skip_reason)
    """
    with h5py.File(ep_path, "r") as ep:
        if "metadata" not in ep:
            raise KeyError(f"{ep_path}: missing 'metadata' group (not a HIL rollout file?)")

        meta = ep["metadata"]
        schema = _attr_str(meta.attrs.get("schema", ""))
        if schema and schema != HIL_SCHEMA:
            raise ValueError(f"{ep_path}: expected schema {HIL_SCHEMA!r}, got {schema!r}")

        end_reason = _attr_str(meta.attrs["end_reason"])
        terminal = bool(meta.attrs["terminal"])
        truncated = bool(meta.attrs["truncated"])
        is_success = classify_rollout_episode(end_reason, terminal, truncated)
        if is_success is None:
            if end_reason == "aborted":
                return {}, torch.empty(0), torch.empty(0), None, None, "aborted"
            return (
                {},
                torch.empty(0),
                torch.empty(0),
                None,
                None,
                f"unmapped end_reason={end_reason!r} terminal={terminal} truncated={truncated}",
            )

        state = torch.from_numpy(np.asarray(ep["action/achieved"][:], dtype=np.float32))
        action = torch.from_numpy(np.asarray(ep["action/commanded"][:], dtype=np.float32))
        num_frames = state.shape[0]
        if num_frames == 0:
            return {}, state, action, None, is_success, "empty"
        if action.shape != (num_frames, STATE_DIM):
            raise ValueError(
                f"{ep_path}: action/commanded shape {tuple(action.shape)} "
                f"!= ({num_frames}, {STATE_DIM})"
            )

        imgs_per_cam: dict[str, np.ndarray] = {}
        for cam in CAMERAS:
            key = f"observation/image_{cam}/jpeg"
            if key not in ep:
                raise KeyError(f"{ep_path}: missing camera dataset {key}")
            ds = ep[key]
            jpegs = [bytes(ds[i]) if ds[i] is not None else b"" for i in range(num_frames)]
            imgs_per_cam[cam] = _decode_jpeg_list_hold_last(jpegs)

        instruction = _attr_str(meta.attrs.get("task_description", "")).strip() or None

    return imgs_per_cam, state, action, instruction, is_success, ""


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    dataset_root: Path,
    progress: ConversionProgress,
    default_task: str,
    success_only: bool = True,
) -> LeRobotDataset:
    skip_counts = {
        "aborted": int(progress.skip_counts.get("aborted", 0)),
        "unmapped": int(progress.skip_counts.get("unmapped", 0)),
        "empty": int(progress.skip_counts.get("empty", 0)),
        "non_success": int(progress.skip_counts.get("non_success", 0)),
    }
    converted = dataset.meta.total_episodes

    for ep_path in tqdm.tqdm(hdf5_files, desc="Episodes"):
        imgs_per_cam, state, action, instruction, is_success, skip_reason = load_episode(ep_path)

        if is_success is None:
            if skip_reason == "aborted":
                skip_counts["aborted"] += 1
            elif skip_reason == "empty":
                skip_counts["empty"] += 1
            else:
                skip_counts["unmapped"] += 1
            tqdm.tqdm.write(f"skipping {ep_path.name} [{skip_reason}]")
            mark_source_processed(
                dataset_root,
                progress,
                ep_path.name,
                skip_reason=skip_reason or "unmapped",
            )
            continue

        if success_only and not is_success:
            skip_counts["non_success"] += 1
            tqdm.tqdm.write(f"skipping {ep_path.name} [non-success]")
            mark_source_processed(
                dataset_root,
                progress,
                ep_path.name,
                skip_reason="non_success",
            )
            continue

        num_frames = state.shape[0]
        for cam in CAMERAS:
            if imgs_per_cam[cam].shape[0] != num_frames:
                raise ValueError(
                    f"{ep_path}: camera {cam} has {imgs_per_cam[cam].shape[0]} "
                    f"frames but state has {num_frames}"
                )
        if action.shape[0] != num_frames:
            raise ValueError(
                f"{ep_path}: action has {action.shape[0]} frames but state has {num_frames}"
            )

        task = instruction if instruction else default_task
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        converted += 1
        mark_source_processed(dataset_root, progress, ep_path.name)

    if converted == 0 and not hdf5_files:
        print("All rollout files already converted.")
    elif converted == dataset.meta.total_episodes and hdf5_files:
        pass
    elif converted == 0:
        raise RuntimeError(
            "No episodes were converted. "
            f"Skipped: {skip_counts['aborted']} aborted, "
            f"{skip_counts['unmapped']} unmapped, "
            f"{skip_counts['empty']} empty, "
            f"{skip_counts['non_success']} non-success."
        )

    print(
        f"Dataset has {converted} episode(s) total. "
        f"Skipped {skip_counts['aborted']} aborted, "
        f"{skip_counts['unmapped']} unmapped, "
        f"{skip_counts['empty']} empty, "
        f"{skip_counts['non_success']} non-success."
    )
    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str = "pick towel from pile, fold and stack",
    episodes: list[int] | None = None,
    max_episodes: int | None = None,
    resume: bool = False,
    success_only: bool = True,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert PiperX HIL rollout HDF5 files to a LeRobot v2.0 dataset.

    Args:
        raw_dir: Directory containing episode_*.hdf5 HIL rollout files.
        repo_id: LeRobot repo id, e.g. "Ishan-Axibo/piperx_flatten_success_rollouts".
        task: Fallback task string when metadata/task_description is missing.
        episodes: Optional list of indices into the sorted file list (debug/smoke test).
        max_episodes: Convert at most this many source files (must match on --resume).
        resume: Continue a previous conversion; skips already-processed HDF5 files.
        success_only: If True, skip failure/timeout episodes (keep success only).
        push_to_hub: If True, also push to Hugging Face Hub.
    """
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {raw_dir}")
    if episodes is not None:
        hdf5_files = [hdf5_files[i] for i in episodes]
    elif max_episodes is not None:
        if max_episodes < 1:
            raise ValueError(f"max_episodes must be >= 1, got {max_episodes}")
        hdf5_files = hdf5_files[:max_episodes]
    print(f"Found {len(hdf5_files)} rollout file(s) to consider in {raw_dir}")

    dataset, target, progress, resumed = open_conversion_dataset(
        repo_id,
        raw_dir=raw_dir,
        max_episodes=max_episodes if episodes is None else len(hdf5_files),
        resume=resume,
        dataset_config=dataset_config,
    )
    progress = ensure_conversion_progress(
        target,
        raw_dir=raw_dir,
        repo_id=repo_id,
        max_episodes=max_episodes if episodes is None else len(hdf5_files),
        existing=progress,
    )
    pending = filter_pending_hdf5(hdf5_files, progress)
    if resumed:
        print(f"Skipping {len(hdf5_files) - len(pending)} already-processed file(s).")
    if not pending:
        print("All rollout files already converted.")
    else:
        print(f"Converting {len(pending)} rollout file(s).")
    dataset = populate_dataset(
        dataset,
        pending,
        dataset_root=target,
        progress=progress,
        default_task=task,
        success_only=success_only,
    )

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print(f"\nLocal dataset written to {Path(LEROBOT_HOME / repo_id).resolve()}")
        print("Skipped Hub push (pass --push-to-hub to enable).")


if __name__ == "__main__":
    tyro.cli(main)
