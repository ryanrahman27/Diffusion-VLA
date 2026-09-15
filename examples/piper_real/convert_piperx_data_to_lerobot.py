"""
Convert PiperX bimanual HDF5 episodes (recorded by episode_recorder.py from
the Agilex Piper teleop stack) into the LeRobot v2.0 dataset format that
openpi consumes.

What this script does:
    * Reads /observations/qpos as observation.state  (14-dim float32)
    * Reads /action as action                        (14-dim float32)
    * Decodes the three JPEG camera streams (cam_front, cam_left_wrist,
      cam_right_wrist) into uint8 image arrays (HWC)
    * Reads /language_instruction as the per-frame "task" string that
      LeRobot 0.1.0 requires
    * Skips depth, qvel, effort, eef_*, time_stamp, sync_time,
      base_action -- openpi does not consume these and including them
      would inflate the dataset.

State / action layout (14-dim, both):
    [L_arm_6 (rad), L_grip (m), R_arm_6 (rad), R_grip (m)]

Gripper values are stored as raw linear position in meters (0.0 = closed,
~0.07 = fully open on the PiperX puppet). Normalization to 0-1 happens
later, inside PiperXInputs at training/inference time -- keeping the
dataset "raw" means we can change normalization choices without
re-converting.

Usage (single episode smoke test):
    uv run python examples/piper_real/convert_piperx_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/folding \\
        --repo-id Ishan-Axibo/piperx_fold_test \\
        --task "fold towel" \\
        --episodes 0

Usage (full conversion):
    uv run python examples/piper_real/convert_piperx_data_to_lerobot.py \\
        --raw-dir /home/axibo/piperx_lerobot_setup/episodes/folding \\
        --repo-id Ishan-Axibo/piperx_fold \\
        --task "fold towel"
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


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

# PiperX has 3 cameras (vs ALOHA's 4). They map cleanly onto pi0's image
# slots: cam_front -> base_0_rgb, cam_left_wrist -> left_wrist_0_rgb,
# cam_right_wrist -> right_wrist_0_rgb. The renaming happens later, in
# PiperXInputs; here we keep the recorder's native names.
CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")

# State / action dimension labels. Metadata only; openpi uses indices.
MOTORS = [
    "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6",
    "right_gripper",
]

# Recording rate from episode_recorder.py default. If you ever re-record
# at a different rate, update this AND the deployment max_hz.
FPS = 30
EEF_6D_KEY = "/observations/eef_6d"
EEF_6D_DIM = 20


def create_empty_dataset(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    include_eef_6d: bool = False,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    if include_eef_6d:
        features["observation.eef_6d"] = {
            "dtype": "float32",
            "shape": (EEF_6D_DIM,),
            "names": [["eef_6d"]],
        }

    # LeRobot 0.1.0 strictly validates image shape, and its writer/loader
    # operate in HWC. Schema must match what we hand to add_frame, which
    # is what cv2.imdecode/cvtColor produce (HWC uint8).
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
    *,
    include_eef_6d: bool = False,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str | None, torch.Tensor | None]:
    """Read one PiperX HDF5 episode.

    Returns:
        (images_per_cam, state, action, instruction, eef_6d_or_none)
        where images_per_cam[cam] has shape (T, H, W, 3) uint8 (RGB).
    """
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(np.asarray(ep["/observations/qpos"][:], dtype=np.float32))
        action = torch.from_numpy(np.asarray(ep["/action"][:], dtype=np.float32))
        eef_6d = None
        if include_eef_6d:
            if EEF_6D_KEY not in ep:
                raise KeyError(
                    f"{ep_path}: missing {EEF_6D_KEY} — required when --include-eef-6d is set."
                )
            eef_6d = torch.from_numpy(np.asarray(ep[EEF_6D_KEY][:], dtype=np.float32))
            if eef_6d.shape[-1] != EEF_6D_DIM:
                raise ValueError(
                    f"{ep_path}: expected {EEF_6D_DIM}-D eef_6d, got shape {tuple(eef_6d.shape)}"
                )

        # Decode JPEG-compressed images per camera. Each entry is a vlen
        # uint8 dataset of JPEG byte strings.
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
            imgs_per_cam[cam] = np.stack(decoded, axis=0)  # (T, H, W, C) uint8

        # Read the per-episode language instruction.
        instruction = None
        if "language_instruction" in ep:
            raw = ep["language_instruction"][()]
            if isinstance(raw, bytes):
                instruction = raw.decode("utf-8")
            else:
                instruction = str(raw)
            instruction = instruction.strip()

    return imgs_per_cam, state, action, instruction, eef_6d


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    default_task: str,
    episodes: list[int] | None = None,
    include_eef_6d: bool = False,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes"):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action, instruction, eef_6d = load_episode(
            ep_path,
            include_eef_6d=include_eef_6d,
        )
        num_frames = state.shape[0]

        # Validate that all streams have the same length. The recorder
        # uses sticky-hold + duplicate-last-row to keep streams aligned,
        # so anything else here is a real bug.
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
        if include_eef_6d and eef_6d is not None and eef_6d.shape[0] != num_frames:
            raise ValueError(
                f"{ep_path}: eef_6d has {eef_6d.shape[0]} frames but state has {num_frames}"
            )

        # Per-episode prompt resolved once, then passed on every frame.
        # LeRobot 0.1.0 requires "task" as a per-frame field, not a
        # per-episode argument to save_episode().
        task = instruction if instruction else default_task

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }
            if include_eef_6d and eef_6d is not None:
                frame["observation.eef_6d"] = eef_6d[i]
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
    include_eef_6d: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert PiperX HDF5 demos to a LeRobot v2.0 dataset.

    Args:
        raw_dir: Directory containing episode_*.hdf5 files.
        repo_id: LeRobot repo id, e.g. "Ishan-Axibo/piperx_fold".
        task: Fallback task string when an episode's /language_instruction
            field is missing or empty.
        episodes: Optional list of indices to convert (otherwise all).
        push_to_hub: If True, also push to Hugging Face Hub.
    """
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
    if include_eef_6d:
        print("Including observation.eef_6d for joint+rel EE training.")

    dataset = create_empty_dataset(
        repo_id,
        dataset_config=dataset_config,
        include_eef_6d=include_eef_6d,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        default_task=task,
        episodes=episodes,
        include_eef_6d=include_eef_6d,
    )

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print(f"\nLocal dataset written to {Path(LEROBOT_HOME / repo_id).resolve()}")
        print("Skipped Hub push (pass --push-to-hub to enable).")


if __name__ == "__main__":
    tyro.cli(main)