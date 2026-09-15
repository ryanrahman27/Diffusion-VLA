"""
Convert PiperX bimanual HDF5 episodes (recorded by episode_recorder.py with
``--depth``) into the LeRobot v2.0 dataset format.

Compared to ``convert_piperx_data_to_lerobot.py``, this script also exports
RealSense Z16 depth and resizes **all** camera streams to 224x224 before writing
the dataset:

    * observation.state  <- /observations/qpos           (14-dim float32)
    * action             <- /action                      (14-dim float32)
    * observation.images.<cam>  <- JPEG RGB, resize-with-pad to 224x224 (matches openpi)
    * observation.depth.<cam>     <- Z16 uint16 PNG (I;16), same pad geometry, nearest resize

Depth is stored as 16-bit grayscale PNGs (raw Z16 values preserved).
Meters = pixel_value * depth_scale_m; scale is read from ``/observations/depth``
group attrs (default 0.001).

Requires HDF5 files recorded with depth enabled (``has_depth=True`` and
``/observations/depth/<cam>`` present). Episodes without depth are skipped
with a warning unless ``--require-depth`` is set (default), in which case
conversion aborts.

Usage (single episode smoke test):
    uv run python examples/piper_real/convert_piperx_data_with_depth_to_lerobot.py \\
        --raw-dir /path/to/episodes/depth \\
        --repo-id Ishan-Axibo/piperx_fold_depth \\
        --task "fold towel" \\
        --episodes 0

Usage (full conversion):
    uv run python examples/piper_real/convert_piperx_data_with_depth_to_lerobot.py \\
        --raw-dir /path/to/episodes/depth \\
        --repo-id Ishan-Axibo/piperx_fold_depth \\
        --task "fold towel"
"""

import dataclasses
from pathlib import Path
import shutil

import cv2
import h5py
from PIL import Image
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import torch
import tqdm
import tyro

from openpi.shared import image_tools


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 3
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")

MOTORS = [
    "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6",
    "right_gripper",
]

FPS = 30
IMAGE_SIZE = 224
DEFAULT_DEPTH_SCALE_M = 0.001


@dataclasses.dataclass(frozen=True)
class ImageSpec:
    height: int = IMAGE_SIZE
    width: int = IMAGE_SIZE


def _resize_rgb_with_pad(img: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """Resize RGB uint8 (H, W, 3) with openpi's resize_with_pad (black padding)."""
    if img.shape[0] == spec.height and img.shape[1] == spec.width:
        return img
    tensor = torch.from_numpy(img)
    padded = image_tools.resize_with_pad_torch(tensor, spec.height, spec.width, mode="bilinear")
    return padded.numpy()


def _resize_depth_with_pad(depth: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """Resize Z16 uint16 (H, W) with the same pad geometry as resize_with_pad."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth.shape}")
    if depth.shape[0] == spec.height and depth.shape[1] == spec.width:
        return depth

    sh, sw = depth.shape[0], depth.shape[1]
    ratio = max(sw / spec.width, sh / spec.height)
    resized_h = int(sh / ratio)
    resized_w = int(sw / ratio)
    resized = cv2.resize(depth, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)

    pad_h0, remainder_h = divmod(spec.height - resized_h, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(spec.width - resized_w, 2)
    pad_w1 = pad_w0 + remainder_w
    return np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1)), mode="constant", constant_values=0)


def _depth_to_pil_image(depth: np.ndarray) -> Image.Image:
    """Wrap Z16 depth (H, W) as a 16-bit grayscale PIL image for PNG export."""
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return Image.fromarray(depth, mode="I;16")


def _episode_has_depth(ep: h5py.File) -> bool:
    if not ep.attrs.get("has_depth", False):
        return False
    depth_grp = ep.get("observations/depth")
    if depth_grp is None:
        return False
    return any(cam in depth_grp for cam in CAMERAS)


def _read_depth_scale(ep: h5py.File) -> float:
    depth_grp = ep.get("observations/depth")
    if depth_grp is None:
        return DEFAULT_DEPTH_SCALE_M
    return float(depth_grp.attrs.get("depth_scale_m", DEFAULT_DEPTH_SCALE_M))


def _count_existing_episodes(repo_id: str) -> int:
    episodes_path = Path(LEROBOT_HOME / repo_id) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return 0
    return sum(1 for line in episodes_path.read_text().splitlines() if line.strip())


def open_dataset_for_resume(
    repo_id: str,
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Re-open a partially written dataset so conversion can continue appending episodes."""
    target = Path(LEROBOT_HOME / repo_id)
    if not target.exists():
        raise FileNotFoundError(f"Cannot resume: dataset does not exist at {target}")

    num_existing = _count_existing_episodes(repo_id)
    if num_existing == 0:
        raise ValueError(f"Cannot resume: no episodes found in {target / 'meta' / 'episodes.jsonl'}")

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.repo_id = repo_id
    dataset.root = target
    dataset.meta = LeRobotDatasetMetadata(repo_id, dataset.root)
    dataset.revision = None
    dataset.tolerance_s = dataset_config.tolerance_s
    dataset.image_writer = None
    dataset.episodes = None
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    from lerobot.common.datasets.video_utils import get_safe_default_codec

    dataset.video_backend = dataset_config.video_backend or get_safe_default_codec()
    dataset.hf_dataset = dataset.load_hf_dataset()
    dataset.episode_buffer = dataset.create_episode_buffer()

    if dataset_config.image_writer_processes or dataset_config.image_writer_threads:
        dataset.start_image_writer(
            dataset_config.image_writer_processes,
            dataset_config.image_writer_threads,
        )

    print(f"Resuming dataset at {target} ({num_existing} episodes already written)")
    return dataset


def create_empty_dataset(
    repo_id: str,
    *,
    image_spec: ImageSpec = ImageSpec(),
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
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

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if dataset_config.use_videos else "image",
            "shape": (image_spec.height, image_spec.width, 3),
            "names": ["height", "width", "channels"],
        }
        features[f"observation.depth.{cam}"] = {
            "dtype": "image",
            "shape": (image_spec.height, image_spec.width, 1),
            "names": ["height", "width", "channels"],
        }

    target = Path(LEROBOT_HOME / repo_id)
    if target.exists():
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="piperx_bimanual_depth",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    default_task: str,
    image_spec: ImageSpec,
    episodes: list[int] | None = None,
    require_depth: bool = True,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    depth_scales: list[float] = []

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes"):
        ep_path = hdf5_files[ep_idx]
        try:
            with h5py.File(ep_path, "r") as ep:
                if not _episode_has_depth(ep):
                    raise ValueError(f"{ep_path}: missing depth (record with episode_recorder.py --depth)")

                depth_scale_m = _read_depth_scale(ep)
                state = torch.from_numpy(np.asarray(ep["/observations/qpos"][:], dtype=np.float32))
                action = torch.from_numpy(np.asarray(ep["/action"][:], dtype=np.float32))
                num_frames = state.shape[0]

                if action.shape[0] != num_frames:
                    raise ValueError(
                        f"{ep_path}: action has {action.shape[0]} frames but state has {num_frames}"
                    )

                depth_grp = ep["observations/depth"]
                image_datasets: dict[str, h5py.Dataset] = {}
                depth_datasets: dict[str, h5py.Dataset] = {}
                for cam in CAMERAS:
                    img_key = f"/observations/images/{cam}"
                    depth_key = f"observations/depth/{cam}"
                    if img_key not in ep:
                        raise KeyError(f"{ep_path}: missing camera dataset {img_key}")
                    if cam not in depth_grp:
                        raise KeyError(f"{ep_path}: missing depth dataset {depth_key}")

                    image_ds = ep[img_key]
                    depth_ds = depth_grp[cam]
                    if image_ds.shape[0] != num_frames:
                        raise ValueError(
                            f"{ep_path}: camera {cam} has {image_ds.shape[0]} "
                            f"frames but state has {num_frames}"
                        )
                    if depth_ds.shape[0] != num_frames:
                        raise ValueError(
                            f"{ep_path}: depth {cam} has {depth_ds.shape[0]} "
                            f"frames but state has {num_frames}"
                        )
                    image_datasets[cam] = image_ds
                    depth_datasets[cam] = depth_ds

                instruction = None
                if "language_instruction" in ep:
                    raw = ep["language_instruction"][()]
                    if isinstance(raw, bytes):
                        instruction = raw.decode("utf-8")
                    else:
                        instruction = str(raw)
                    instruction = instruction.strip()

                task = instruction if instruction else default_task

                for i in range(num_frames):
                    frame = {
                        "observation.state": state[i],
                        "action": action[i],
                        "task": task,
                    }
                    for cam in CAMERAS:
                        jpeg_bytes = image_datasets[cam][i]
                        arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
                        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            raise ValueError(f"{ep_path}: failed to decode a JPEG in {cam}")
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        frame[f"observation.images.{cam}"] = _resize_rgb_with_pad(img_rgb, image_spec)

                        depth_raw = np.asarray(depth_datasets[cam][i], dtype=np.uint16)
                        if depth_raw.ndim != 2:
                            raise ValueError(
                                f"{ep_path}: expected depth/{cam} frame shape (H, W), got {depth_raw.shape}"
                            )
                        depth_resized = _resize_depth_with_pad(depth_raw, image_spec)
                        frame[f"observation.depth.{cam}"] = _depth_to_pil_image(depth_resized)

                    dataset.add_frame(frame)
        except ValueError as exc:
            if require_depth:
                raise
            print(f"Skipping {ep_path.name}: {exc}")
            continue

        depth_scales.append(depth_scale_m)
        dataset.save_episode()

    if depth_scales:
        unique_scales = sorted(set(depth_scales))
        if len(unique_scales) > 1:
            print(
                f"Warning: depth_scale_m varied across episodes: {unique_scales}. "
                "Depth PNGs store raw Z16; convert to meters per-episode if scales differ."
            )
        else:
            print(f"Depth scale: {unique_scales[0]} m per Z16 unit (meters = PNG pixel * scale)")

    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str = "fold towel",
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    image_size: int = IMAGE_SIZE,
    require_depth: bool = True,
    resume: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert PiperX HDF5 demos (with depth) to a LeRobot v2.0 dataset at 224x224.

    Args:
        raw_dir: Directory containing episode_*.hdf5 files recorded with --depth.
        repo_id: LeRobot repo id, e.g. "Ishan-Axibo/piperx_fold_depth".
        task: Fallback task string when an episode's /language_instruction is missing.
        episodes: Optional list of indices to convert (otherwise all).
        push_to_hub: If True, also push to Hugging Face Hub.
        image_size: Square resolution for RGB and depth (default 224).
        require_depth: If True, abort on episodes without depth; else skip them.
        resume: If True, continue appending to an existing local dataset instead of deleting it.
    """
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")

    image_spec = ImageSpec(height=image_size, width=image_size)

    target = Path(LEROBOT_HOME / repo_id)
    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *.hdf5 files found in {raw_dir}")
    print(f"Found {len(hdf5_files)} episodes in {raw_dir}")
    print(f"Output resolution: {image_size}x{image_size} (resize-with-pad), RGB as video={dataset_config.use_videos}")

    if resume:
        if not target.exists():
            raise FileNotFoundError(f"Cannot resume: dataset does not exist at {target}")
        start_idx = _count_existing_episodes(repo_id)
        if start_idx >= len(hdf5_files):
            print(f"Dataset already has {start_idx} episodes; nothing left to convert.")
            return
        if episodes is None:
            episodes = list(range(start_idx, len(hdf5_files)))
        print(f"Resuming from HDF5 episode index {episodes[0]} ({start_idx} already converted)")
        dataset = open_dataset_for_resume(repo_id, dataset_config=dataset_config)
    else:
        if target.exists():
            print(f"Removing existing dataset at {target}")
            shutil.rmtree(target)
        if episodes is None:
            episodes = list(range(len(hdf5_files)))
        dataset = create_empty_dataset(repo_id, image_spec=image_spec, dataset_config=dataset_config)

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        default_task=task,
        image_spec=image_spec,
        episodes=episodes,
        require_depth=require_depth,
    )

    if push_to_hub:
        dataset.push_to_hub()
    else:
        print(f"\nLocal dataset written to {Path(LEROBOT_HOME / repo_id).resolve()}")
        print("Skipped Hub push (pass --push-to-hub to enable).")


if __name__ == "__main__":
    tyro.cli(main)
