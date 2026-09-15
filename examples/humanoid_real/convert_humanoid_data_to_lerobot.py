"""
Convert humanoid upper-body HDF5 episodes (recorded by episode_recorder.py) into
LeRobot v2.0 format for openpi training.

Supported HDF5 schemas
----------------------
humanoid_async_v1 (pickV2 and newer — async multi-rate recorder output):
    streams/robot_state/{t,qpos,...}   native rate (~1 kHz)
    streams/hand_{left,right}/{t,pinch_dist,...}
    streams/images/<cam>/{t,jpeg}
    Resampled to 30 Hz at conversion via ``async_episode_io.resample_async_hdf5``.

humanoid_arms_v1 (legacy pick):
    observations/qpos   (T, 16)  14 arm joints + 2 hand pinch distances (m)
    action              (T, 14)  optional; ignored — actions are built from t+1 state

humanoid_arms_v2_omnihand_state (pick_and_place):
    observations/qpos              (T, 38)  arms + full hand joints + pinch fields
    observations/hand_pinch/pinch_dist (T, 2)  left/right thumb-index distance (m)
    no /action dataset

Compact 16-D layout written to LeRobot (all schemas):
    [L_arm_7 (rad), R_arm_7 (rad), L_pinch_dist (m), R_pinch_dist (m)]

Only arm joints and pinch distance are kept — OmniHand motor joints (10 per hand)
are dropped from state/action.

Actions use next-state (t+1) for all 16 dims:
    action[t] = compact_state[t+1]
    action[T-1] = compact_state[T-1]

Environment
-----------
PIPERX_LEROBOT_SETUP   Path to piperx_lerobot_setup repo (for async_episode_io).
                       Default: /home/axibo/piperx_lerobot_setup

Usage (smoke test, one episode):
    uv run python examples/humanoid_real/convert_humanoid_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/pickV2 \\
        --repo-id local/humanoid_pickV2 \\
        --episodes 0

Usage (full conversion):
    uv run python examples/humanoid_real/convert_humanoid_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/pick_and_place \\
        --repo-id local/humanoid_pick_and_place

Usage (laundry folding — pi05_humanoid_laundry_fold):
    uv run python examples/humanoid_real/convert_humanoid_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/laundry \\
        --repo-id local/humanoid_laundry \\
        --task "fold towel"
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")

MOTORS = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    "left_hand_pinch_dist",
    "right_hand_pinch_dist",
]

STATE_DIM = len(MOTORS)  # 16
ARM_DIM = 14
# Indices in the full 38-D v2 qpos vector.
_V2_PINCH_DIST_QPOS_SLICE = slice(34, 36)
FPS = 30

# Recorder placeholder JPEG (1x1) written before first real camera frame.
_PLACEHOLDER_JPEG_LEN = 336

_LEROBOT_SETUP = Path(
    os.environ.get("PIPERX_LEROBOT_SETUP", "/home/axibo/piperx_lerobot_setup")
).resolve()
_SCRIPTS_DIR = _LEROBOT_SETUP / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from async_episode_io import is_async_hdf5, resample_async_hdf5  # noqa: E402


def _is_placeholder_jpeg(jpeg_bytes: bytes) -> bool:
    return len(jpeg_bytes) == _PLACEHOLDER_JPEG_LEN


def _decode_jpeg_rgb(jpeg_bytes: bytes) -> np.ndarray | None:
    if len(jpeg_bytes) < 4 or _is_placeholder_jpeg(jpeg_bytes):
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _read_pinch_dist(ep: h5py.File, *, num_frames: int) -> np.ndarray:
    """Read left/right pinch distance (metres) from the preferred v2 dataset."""
    key = "/observations/hand_pinch/pinch_dist"
    if key in ep:
        pinch = np.asarray(ep[key][:], dtype=np.float32)
        if pinch.shape != (num_frames, 2):
            raise ValueError(f"{key} shape {pinch.shape} != ({num_frames}, 2)")
        return pinch

    qpos = np.asarray(ep["/observations/qpos"][:], dtype=np.float32)
    if qpos.shape[1] >= 36:
        return qpos[:, _V2_PINCH_DIST_QPOS_SLICE].astype(np.float32)
    if qpos.shape[1] == STATE_DIM:
        return qpos[:, ARM_DIM:STATE_DIM].astype(np.float32)

    raise ValueError(
        "Could not find pinch distance: expected "
        "/observations/hand_pinch/pinch_dist or qpos with >= 36 dims"
    )


def extract_compact_state_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Build (T, 16) from a qpos array (16-D legacy or 38-D v2/async resampled)."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < ARM_DIM:
        raise ValueError(f"qpos must be (T, >=14), got {qpos.shape}")

    arms = qpos[:, :ARM_DIM]
    if qpos.shape[1] >= 36:
        pinch = qpos[:, _V2_PINCH_DIST_QPOS_SLICE].astype(np.float32)
    elif qpos.shape[1] == STATE_DIM:
        pinch = qpos[:, ARM_DIM:STATE_DIM].astype(np.float32)
    else:
        raise ValueError(
            f"qpos has {qpos.shape[1]} dims; need 16 (legacy) or >=36 (v2/async)"
        )
    return np.concatenate([arms, pinch], axis=-1).astype(np.float32)


def extract_compact_state(ep: h5py.File) -> np.ndarray:
    """Build (T, 16) state from sync HDF5 (v1/v2 pre-resampled episodes)."""
    qpos = np.asarray(ep["/observations/qpos"][:], dtype=np.float32)
    if qpos.shape[1] >= 36:
        return extract_compact_state_from_qpos(qpos)
    pinch = _read_pinch_dist(ep, num_frames=qpos.shape[0])
    arms = qpos[:, :ARM_DIM]
    return np.concatenate([arms, pinch], axis=-1).astype(np.float32)


def build_next_state_actions(state: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1]; last frame repeats final state."""
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


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

    target = Path(LEROBOT_HOME / repo_id)
    if target.exists():
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="humanoid_upper_body",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _read_instruction(ep: h5py.File) -> str | None:
    if "language_instruction" not in ep:
        if "instruction" in ep.attrs:
            raw = ep.attrs["instruction"]
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace").strip()
            return str(raw).strip()
        return None
    raw = ep["language_instruction"][()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def _decode_jpeg_sequence(
    ep_path: Path,
    cam: str,
    jpeg_frames: list[bytes] | h5py.Dataset,
) -> np.ndarray:
    decoded: list[np.ndarray] = []
    last_good: np.ndarray | None = None
    for frame_idx, jpeg_bytes in enumerate(jpeg_frames):
        img = _decode_jpeg_rgb(bytes(jpeg_bytes))
        if img is None:
            if last_good is not None:
                img = last_good
            else:
                raise ValueError(
                    f"{ep_path}: failed to decode a JPEG in {cam} at frame {frame_idx} "
                    f"and no prior good frame to reuse"
                )
        else:
            last_good = img
        decoded.append(img)
    return np.stack(decoded, axis=0)


def load_episode_sync(
    ep_path: Path,
    ep: h5py.File,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None]:
    state_np = extract_compact_state(ep)
    action_np = build_next_state_actions(state_np)
    state = torch.from_numpy(state_np)
    action = torch.from_numpy(action_np)

    imgs_per_cam = {}
    for cam in CAMERAS:
        key = f"/observations/images/{cam}"
        if key not in ep:
            raise KeyError(f"{ep_path}: missing camera dataset {key}")
        imgs_per_cam[cam] = _decode_jpeg_sequence(ep_path, cam, ep[key])

    return imgs_per_cam, state, action, _read_instruction(ep)


def load_episode_async(
    ep_path: Path,
    ep: h5py.File,
    *,
    fps: float = FPS,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None]:
    res = resample_async_hdf5(ep, fps=fps)
    state_np = extract_compact_state_from_qpos(res.qpos)
    action_np = build_next_state_actions(state_np)
    state = torch.from_numpy(state_np)
    action = torch.from_numpy(action_np)

    imgs_per_cam = {}
    for cam in CAMERAS:
        if cam not in res.cam_jpeg:
            raise KeyError(f"{ep_path}: resampled episode missing camera {cam}")
        imgs_per_cam[cam] = _decode_jpeg_sequence(ep_path, cam, res.cam_jpeg[cam])

    return imgs_per_cam, state, action, _read_instruction(ep)


def load_episode(
    ep_path: Path,
    *,
    fps: float = FPS,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None]:
    with h5py.File(ep_path, "r") as ep:
        if is_async_hdf5(ep):
            return load_episode_async(ep_path, ep, fps=fps)
        return load_episode_sync(ep_path, ep)


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    default_task: str,
    episodes: list[int] | None = None,
    fps: float = FPS,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes"):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action, instruction = load_episode(ep_path, fps=fps)
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

    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str = "pick up cube and place in bin",
    episodes: list[int] | None = None,
    fps: float = FPS,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *.hdf5 files found in {raw_dir}")
    print(f"Found {len(hdf5_files)} episodes in {raw_dir}")

    dataset = create_empty_dataset(repo_id, dataset_config=dataset_config)
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        default_task=task,
        episodes=episodes,
        fps=fps,
    )

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print(f"\nLocal dataset written to {Path(LEROBOT_HOME / repo_id).resolve()}")
        print("Skipped Hub push (pass --push-to-hub to enable).")


if __name__ == "__main__":
    tyro.cli(main)
