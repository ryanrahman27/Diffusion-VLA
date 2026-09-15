"""
Convert UMI-style single-arm EE HDF5 episodes (``umi_ee_v1``) to LeRobot format.

HDF5 schema (from ``umi_ee`` recorder):
    * observation.state  <- ee_pose[t] + gripper_width[t]   (10-D rot6d EE)
    * action             <- ee_pose[t+1] + gripper_width[t+1]
    * observation.images.cam_left_wrist <- observations/images/wrist[t]

State / action layout (10-dim):
    [x, y, z, rot6d_0..5, gripper (m)]

Usage:
    uv run python examples/piper_real/ee/convert_umi_ee_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/umi_ee \\
        --repo-id axiboai/piper-umi-ee \\
        --task "pick cube and place in bin"

Then:
    uv run scripts/compute_norm_stats.py --config-name pi05_piper_umi_ee_rel_traj
    uv run scripts/train.py pi05_piper_umi_ee_rel_traj --exp-name=umi_ee_v1 --overwrite
"""

import dataclasses
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

from openpi.policies.piperx_rel_pose import rotation_matrix_to_rot6d

EE_DIM = 10
CAMERA = "cam_left_wrist"
WRIST_IMAGE_KEY = "observations/images/wrist"
EE_POSE_KEY = "observations/ee_pose"
GRIPPER_WIDTH_KEY = "observations/gripper_width"

EE_DIM_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
]

FPS = 30


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def ee_pose_and_gripper_to_state(ee_pose: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
    """Convert (T, 4, 4) poses + (T,) gripper widths to (T, 10) eef_6d vectors."""
    ee_pose = np.asarray(ee_pose, dtype=np.float32)
    gripper_width = np.asarray(gripper_width, dtype=np.float32)
    xyz = ee_pose[..., :3, 3]
    rotation = ee_pose[..., :3, :3]
    if rotation.ndim == 2:
        rot6d = rotation_matrix_to_rot6d(rotation)
        return np.concatenate([xyz, rot6d, gripper_width.reshape(1)], axis=-1).astype(np.float32)
    rot6d = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=-1).astype(np.float32)
    return np.concatenate([xyz, rot6d, gripper_width[..., None]], axis=-1).astype(np.float32)


def _next_step_actions(state: np.ndarray) -> np.ndarray:
    if state.shape[0] == 0:
        return state
    if state.shape[0] == 1:
        return state.copy()
    actions = np.empty_like(state)
    actions[:-1] = state[1:]
    actions[-1] = state[-1]
    return actions


def _decode_instruction(ep: h5py.File) -> str | None:
    if "language_instruction" in ep:
        raw = ep["language_instruction"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        text = text.strip()
        if text:
            return text
    task_label = ep.attrs.get("task_label")
    if task_label is not None:
        text = str(task_label).strip().replace("_", " ")
        if text:
            return text
    return None


def create_empty_dataset(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (EE_DIM,),
            "names": [EE_DIM_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (EE_DIM,),
            "names": [EE_DIM_NAMES],
        },
        f"observation.images.{CAMERA}": {
            "dtype": "video" if dataset_config.use_videos else "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }

    target = Path(LEROBOT_HOME / repo_id)
    if target.exists():
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="piper_single_eef_6d",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_episode(ep_path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, str | None]:
    with h5py.File(ep_path, "r") as ep:
        schema = ep.attrs.get("schema")
        if schema is not None and str(schema) != "umi_ee_v1":
            raise ValueError(f"{ep_path}: expected schema umi_ee_v1, got {schema!r}")

        for key in (EE_POSE_KEY, GRIPPER_WIDTH_KEY, WRIST_IMAGE_KEY):
            if key not in ep:
                raise KeyError(f"{ep_path}: missing dataset {key}")

        ee_pose = np.asarray(ep[EE_POSE_KEY][:], dtype=np.float32)
        gripper_width = np.asarray(ep[GRIPPER_WIDTH_KEY][:], dtype=np.float32)
        if ee_pose.ndim != 3 or ee_pose.shape[1:] != (4, 4):
            raise ValueError(f"{ep_path}: expected ee_pose shape (T, 4, 4), got {ee_pose.shape}")
        if gripper_width.ndim != 1 or gripper_width.shape[0] != ee_pose.shape[0]:
            raise ValueError(
                f"{ep_path}: gripper_width length {gripper_width.shape} "
                f"!= ee_pose frames {ee_pose.shape[0]}"
            )

        eef_6d = ee_pose_and_gripper_to_state(ee_pose, gripper_width)
        state = torch.from_numpy(eef_6d)
        action = torch.from_numpy(_next_step_actions(eef_6d))

        ds = ep[WRIST_IMAGE_KEY]
        decoded = []
        for jpeg_bytes in ds:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"{ep_path}: failed to decode a JPEG in wrist camera")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            decoded.append(img_rgb)
        images = np.stack(decoded, axis=0)

        instruction = _decode_instruction(ep)

    return images, state, action, instruction


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    default_task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes"):
        ep_path = hdf5_files[ep_idx]
        images, state, action, instruction = load_episode(ep_path)
        num_frames = state.shape[0]

        if images.shape[0] != num_frames:
            raise ValueError(
                f"{ep_path}: wrist camera has {images.shape[0]} frames but state has {num_frames}"
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
                f"observation.images.{CAMERA}": images[i],
            }
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str = "pick cube and place in bin",
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")

    target = Path(LEROBOT_HOME / repo_id)
    if target.exists():
        print(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *.hdf5 files found in {raw_dir}")
    print(f"Found {len(hdf5_files)} episodes in {raw_dir}")
    print("EE 6D convention: state=ee_pose[t]+grip[t], action=ee_pose[t+1]+grip[t+1]")

    dataset = create_empty_dataset(repo_id, dataset_config=dataset_config)
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        default_task=task,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print(f"\nLocal dataset written to {Path(LEROBOT_HOME / repo_id).resolve()}")
        print("Skipped Hub push (pass --push-to-hub to enable).")


if __name__ == "__main__":
    tyro.cli(main)
