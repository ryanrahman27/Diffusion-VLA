"""
Convert PiperX HDF5 episodes to LeRobot format using **end-effector 6D** pose.

Uses existing recorder output (no recorder changes):
    * observation.state  <- /observations/eef_6d[t]     (follower EE at t)
    * action             <- /observations/eef_6d[t+1]   (next achieved EE; last frame repeats)

State / action layout (20-dim):
    [L_x, L_y, L_z, L_rot6d_0..5, L_grip (m),
     R_x, R_y, R_z, R_rot6d_0..5, R_grip (m)]

rot6d = first two columns of the rotation matrix (continuous; no RPY wrap).

Usage:
    uv run python examples/piper_real/ee/convert_piperx_ee_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/folding \\
        --repo-id Ishan-Axibo/piperx_fold_ee \\
        --task "fold towel"

Then:
    uv run scripts/compute_norm_stats.py --config-name pi05_piperx_fold_ee
    uv run scripts/train.py pi05_piperx_fold_ee --exp-name=fold_ee_v1 --overwrite
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

EEF_6D_KEY = "/observations/eef_6d"
EEF_6D_DIM = 20


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")

EE_DIM_NAMES = [
    "left_x", "left_y", "left_z",
    "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
    "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
    "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
    "right_gripper",
]

FPS = 30


def create_empty_dataset(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (EEF_6D_DIM,),
            "names": [EE_DIM_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (EEF_6D_DIM,),
            "names": [EE_DIM_NAMES],
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
        robot_type="piperx_bimanual_eef_6d",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _next_step_actions(pose: np.ndarray) -> np.ndarray:
    """action[t] = pose[t+1], last frame repeats."""
    if pose.shape[0] == 0:
        return pose
    if pose.shape[0] == 1:
        return pose.copy()
    actions = np.empty_like(pose)
    actions[:-1] = pose[1:]
    actions[-1] = pose[-1]
    return actions


def load_episode(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None]:
    with h5py.File(ep_path, "r") as ep:
        if EEF_6D_KEY not in ep:
            raise KeyError(
                f"{ep_path}: missing {EEF_6D_KEY} — record with episode_recorder.py "
                "(eef_6d is written automatically alongside eef / qpos)."
            )
        eef_6d = np.asarray(ep[EEF_6D_KEY][:], dtype=np.float32)
        if eef_6d.shape[-1] != EEF_6D_DIM:
            raise ValueError(f"{ep_path}: expected {EEF_6D_DIM}-D eef_6d, got shape {eef_6d.shape}")
        state = torch.from_numpy(eef_6d)
        action = torch.from_numpy(_next_step_actions(eef_6d))

        imgs_per_cam = {}
        for cam in CAMERAS:
            key = f"/observations/images/{cam}"
            if key not in ep:
                raise KeyError(f"{ep_path}: missing camera dataset {key}")
            ds = ep[key]
            decoded = []
            for jpeg_bytes in ds:
                arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"{ep_path}: failed to decode a JPEG in {cam}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                decoded.append(img_rgb)
            imgs_per_cam[cam] = np.stack(decoded, axis=0)

        instruction = None
        if "language_instruction" in ep:
            raw = ep["language_instruction"][()]
            if isinstance(raw, bytes):
                instruction = raw.decode("utf-8")
            else:
                instruction = str(raw)
            instruction = instruction.strip()

    return imgs_per_cam, state, action, instruction


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
        imgs_per_cam, state, action, instruction = load_episode(ep_path)
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
    task: str = "fold towel",
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
    print("EE 6D convention: state=eef_6d[t], action=eef_6d[t+1] (last frame repeats)")

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
