"""Merge multiple LeRobot datasets into one.

openpi trains from a single ``repo_id``; use this to combine datasets before
``compute_norm_stats`` / ``train.py``.

By default this **copies parquet + mp4 files** and reindexes metadata. If camera
resolutions differ between sources, mismatched videos are **ffmpeg-rescaled** to
match the **first** source's image shapes (put the dataset whose resolution you
want as output first — e.g. folding @ 224×224 before stacking @ 640×480).

Pass ``--slow`` to fall back to frame-by-frame decode/re-encode via LeRobot.

Example:
    uv run python examples/piper_real/merge_lerobot_datasets.py \\
        --output-repo-id axiboai/piper_stacking_folding_v1 \\
        --source-repo-ids axiboai/piper_folding_iwr_v1 axiboai/piper_stacking_v3 \\
        --push-to-hub
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import packaging.version
import pandas as pd
import torch
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import write_info, write_stats


def _image_keys(features: dict) -> list[str]:
    return [k for k in features if k.startswith("observation.images.")]


def _non_image_features_match(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        if key in _image_keys(a):
            continue
        for field in ("dtype", "shape"):
            if a[key].get(field) != b[key].get(field):
                return False
    return True


def _image_shapes_match(a: dict, b: dict) -> bool:
    for key in _image_keys(a):
        if a[key].get("shape") != b[key].get("shape"):
            return False
    return True


def _rescale_video(src: Path, dst: Path, *, height: int, width: int) -> None:
    """Rescale an mp4 to ``(height, width)`` with ffmpeg (LeRobot shape is HWC)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale={width}:{height}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-an",
            str(dst),
        ],
        check=True,
    )


def _copy_or_rescale_video(
    src_vid: Path,
    dst_vid: Path,
    *,
    src_shape: list,
    dst_shape: list,
) -> None:
    if src_shape == dst_shape:
        shutil.copy2(src_vid, dst_vid)
        return
    height, width, _ = dst_shape
    _rescale_video(src_vid, dst_vid, height=height, width=width)


def _validate_sources(sources: list[LeRobotDatasetMetadata]) -> LeRobotDatasetMetadata:
    template = sources[0]
    for src in sources[1:]:
        if not _non_image_features_match(template.features, src.features):
            raise ValueError(
                f"Non-image feature mismatch between {template.repo_id} and {src.repo_id}. "
                "State/action keys, dtypes, and shapes must match."
            )
        if not _image_shapes_match(template.features, src.features):
            print(
                f"Camera resolution mismatch: {template.repo_id} "
                f"{template.features[_image_keys(template.features)[0]]['shape']} vs "
                f"{src.repo_id} {src.features[_image_keys(src.features)[0]]['shape']} "
                f"— will ffmpeg-rescale {src.repo_id} to match {template.repo_id}."
            )
        if src.fps != template.fps:
            raise ValueError(f"FPS mismatch: {template.repo_id}={template.fps}, {src.repo_id}={src.fps}")
        if src.chunks_size != template.chunks_size:
            raise ValueError(
                f"chunks_size mismatch: {template.repo_id}={template.chunks_size}, "
                f"{src.repo_id}={src.chunks_size}"
            )
        if src.data_path != template.data_path or src.video_path != template.video_path:
            raise ValueError(
                f"Path template mismatch between {template.repo_id} and {src.repo_id}. "
                "Use --slow for incompatible layouts."
            )
        if src._version != template._version:
            raise ValueError(
                f"LeRobot version mismatch: {template.repo_id}={template.info['codebase_version']}, "
                f"{src.repo_id}={src.info['codebase_version']}. Fast merge requires identical versions."
            )
    if template._version < packaging.version.parse("v2.1"):
        raise ValueError(
            f"Fast merge requires LeRobot v2.1+ (per-episode parquet). "
            f"{template.repo_id} is {template.info['codebase_version']}; use --slow."
        )
    return template


def _ensure_task(dst: LeRobotDatasetMetadata, task: str) -> int:
    task_index = dst.get_task_index(task)
    if task_index is None:
        dst.add_task(task)
        task_index = dst.get_task_index(task)
    assert task_index is not None
    return task_index


def _remap_parquet(
    src_df: pd.DataFrame,
    *,
    src_tasks: dict[int, str],
    dst_episode_index: int,
    dst_frame_offset: int,
    dst: LeRobotDatasetMetadata,
) -> pd.DataFrame:
    df = src_df.copy()
    ep_len = len(df)
    df["episode_index"] = dst_episode_index
    df["index"] = np.arange(dst_frame_offset, dst_frame_offset + ep_len, dtype=np.int64)
    df["frame_index"] = np.arange(ep_len, dtype=np.int64)
    df["timestamp"] = df["frame_index"].to_numpy(dtype=np.float64) / dst.fps
    df["task_index"] = df["task_index"].map(lambda ti: _ensure_task(dst, src_tasks[int(ti)]))
    return df


def _source_files_complete(src: LeRobotDatasetMetadata) -> bool:
    ep_indices = sorted(src.episodes)
    if not ep_indices:
        return True
    sample_parquet = src.root / src.get_data_file_path(ep_indices[0])
    if not sample_parquet.is_file():
        return False
    if src.video_keys:
        sample_video = src.root / src.get_video_file_path(ep_indices[0], src.video_keys[0])
        return sample_video.is_file()
    return True


def _ensure_source_downloaded(repo_id: str) -> LeRobotDatasetMetadata:
    """Ensure parquet + videos exist under LeRobot home (not just HF hub cache)."""
    meta = LeRobotDatasetMetadata(repo_id)
    if _source_files_complete(meta):
        return meta

    print(f"Downloading {repo_id} into {meta.root} ...")
    # LeRobotDataset triggers download_episodes() when files are missing.
    LeRobotDataset(repo_id, revision=meta.revision)
    meta = LeRobotDatasetMetadata(repo_id)
    if _source_files_complete(meta):
        return meta

    # Hub CLI downloads often land in ~/.cache/hub only; materialize into LeRobot home.
    from huggingface_hub import snapshot_download

    print(f"Materializing {repo_id} snapshot into {meta.root} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=meta.revision,
        local_dir=meta.root,
    )
    meta = LeRobotDatasetMetadata(repo_id)
    if not _source_files_complete(meta):
        raise FileNotFoundError(
            f"Dataset files still missing under {meta.root}. "
            f"Run: huggingface-cli download {repo_id} --repo-type dataset "
            f"--local-dir {meta.root}"
        )
    return meta


def _fast_copy_episodes(src: LeRobotDatasetMetadata, dst: LeRobotDatasetMetadata) -> int:
    video_keys = src.video_keys
    count = 0
    for src_ep_idx in tqdm.tqdm(sorted(src.episodes), desc=f"copy {src.repo_id}"):
        ep_info = src.episodes[src_ep_idx]
        dst_ep_idx = dst.total_episodes
        dst_frame_offset = dst.total_frames

        src_parquet = src.root / src.get_data_file_path(src_ep_idx)
        if not src_parquet.is_file():
            raise FileNotFoundError(
                f"Missing {src_parquet}. Hub download failed or dataset is incomplete."
            )
        df = pd.read_parquet(src_parquet)
        df = _remap_parquet(
            df,
            src_tasks=src.tasks,
            dst_episode_index=dst_ep_idx,
            dst_frame_offset=dst_frame_offset,
            dst=dst,
        )
        dst_parquet = dst.root / dst.get_data_file_path(dst_ep_idx)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_parquet, index=False)

        for vid_key in video_keys:
            src_vid = src.root / src.get_video_file_path(src_ep_idx, vid_key)
            dst_vid = dst.root / dst.get_video_file_path(dst_ep_idx, vid_key)
            _copy_or_rescale_video(
                src_vid,
                dst_vid,
                src_shape=src.features[vid_key]["shape"],
                dst_shape=dst.features[vid_key]["shape"],
            )

        ep_stats = src.episodes_stats[src_ep_idx]
        dst.save_episode(dst_ep_idx, ep_info["length"], ep_info["tasks"], ep_stats)
        count += 1
    return count


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_task(item: dict) -> str:
    task = item.get("task")
    if task is None:
        return ""
    if isinstance(task, torch.Tensor):
        if task.ndim == 0:
            return str(task.item())
        return str(task[0])
    if isinstance(task, (list, tuple)) and task:
        return str(task[0])
    return str(task)


def _slow_copy_episodes(src: LeRobotDataset, dst: LeRobotDataset) -> int:
    camera_keys = [k for k in src.meta.features if k.startswith("observation.images.")]
    count = 0
    for ep_idx in tqdm.tqdm(range(src.num_episodes), desc=f"copy {src.repo_id}"):
        ep_start = int(src.episode_data_index["from"][ep_idx].item())
        ep_end = int(src.episode_data_index["to"][ep_idx].item())
        task_str = ""
        for frame_idx in range(ep_start, ep_end):
            item = src[frame_idx]
            if not task_str:
                task_str = _frame_task(item)
            frame = {
                "observation.state": _to_numpy(item["observation.state"]).astype(np.float32),
                "action": _to_numpy(item["action"]).astype(np.float32),
                "task": task_str,
            }
            for key in camera_keys:
                img = item[key]
                if isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                arr = _to_numpy(img)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                frame[key] = arr
            if "observation.eef_6d" in src.meta.features and "observation.eef_6d" in item:
                frame["observation.eef_6d"] = _to_numpy(item["observation.eef_6d"]).astype(np.float32)
            dst.add_frame(frame)
        dst.save_episode()
        count += 1
    return count


def merge_datasets(
    source_repo_ids: tuple[str, ...],
    output_repo_id: str,
    *,
    overwrite: bool = True,
    push_to_hub: bool = False,
    fast: bool = True,
) -> LeRobotDataset:
    if not source_repo_ids:
        raise ValueError("At least one --source-repo-id is required")

    out_path = Path(LEROBOT_HOME / output_repo_id)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite or pick a new repo id")
        shutil.rmtree(out_path)

    if fast:
        source_metas = [_ensure_source_downloaded(repo_id) for repo_id in source_repo_ids]
        template = _validate_sources(source_metas)

        dst_meta = LeRobotDatasetMetadata.create(
            repo_id=output_repo_id,
            fps=template.fps,
            robot_type=template.robot_type,
            features=template.features,
            use_videos=template.video_path is not None,
        )
        dst_meta.info["chunks_size"] = template.chunks_size
        write_info(dst_meta.info, dst_meta.root)

        total_eps = 0
        for src_meta in source_metas:
            n = _fast_copy_episodes(src_meta, dst_meta)
            print(f"Copied {n} episodes from {src_meta.repo_id}")
            total_eps += n

        dst_meta.stats = aggregate_stats(list(dst_meta.episodes_stats.values()))
        write_stats(dst_meta.stats, dst_meta.root)
        print(f"Merged {total_eps} episodes -> {out_path} (fast copy + ffmpeg rescale where needed)")
        merged = LeRobotDataset(output_repo_id)
    else:
        sources = [LeRobotDataset(repo_id) for repo_id in source_repo_ids]
        template_ds = sources[0]
        for src in sources[1:]:
            if src.meta.features.keys() != template_ds.meta.features.keys():
                raise ValueError(
                    f"Feature mismatch between {template_ds.repo_id} and {src.repo_id}. "
                    "Convert ABC with convert_abc_mcap_to_lerobot.py (PiperX schema)."
                )
            if src.fps != template_ds.fps:
                raise ValueError(f"FPS mismatch: {template_ds.repo_id}={template_ds.fps}, {src.repo_id}={src.fps}")

        dst = LeRobotDataset.create(
            repo_id=output_repo_id,
            fps=template_ds.fps,
            robot_type=template_ds.meta.robot_type,
            features=template_ds.meta.features,
            use_videos=True,
            tolerance_s=template_ds.tolerance_s,
        )
        total_eps = 0
        for src in sources:
            n = _slow_copy_episodes(src, dst)
            print(f"Copied {n} episodes from {src.repo_id}")
            total_eps += n
        print(f"Merged {total_eps} episodes -> {out_path} (slow decode/re-encode)")
        merged = dst

    if push_to_hub:
        merged.push_to_hub()
    return merged


def main(
    output_repo_id: str,
    source_repo_ids: tuple[str, ...],
    *,
    overwrite: bool = True,
    push_to_hub: bool = False,
    slow: bool = False,
) -> None:
    merge_datasets(
        source_repo_ids,
        output_repo_id,
        overwrite=overwrite,
        push_to_hub=push_to_hub,
        fast=not slow,
    )


if __name__ == "__main__":
    tyro.cli(main)
