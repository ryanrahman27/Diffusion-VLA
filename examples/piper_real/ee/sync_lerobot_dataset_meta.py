"""Rebuild LeRobot meta from files on disk (fixes timestamp sync errors).

Run this after removing corrupt episodes or if training fails with errors like:
  {'diff': -56.53, 'episode_index': 239, 'timestamps': [56.53, 0.0]}

That happens when meta/episodes.jsonl lists episodes/lengths that don't match
the parquet files actually present (e.g. episode 153 removed from disk but
still listed in episodes.jsonl).

Usage:
    export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot   # if needed

    uv run python examples/piper_real/ee/sync_lerobot_dataset_meta.py \\
        --repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
import pyarrow.parquet as pq
import tyro


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def sync_dataset_meta(
    *,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    root = Path(LEROBOT_HOME / repo_id)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    tasks_path = root / "meta" / "tasks.jsonl"
    task_rows = _read_jsonl(tasks_path)
    task_index_to_name = {row["task_index"]: row["task"] for row in task_rows}

    old_episodes = {row["episode_index"]: row for row in _read_jsonl(root / "meta" / "episodes.jsonl")}
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    new_episodes: list[dict] = []
    total_frames = 0
    missing_videos: list[tuple[int, str]] = []

    video_keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    n_cams = len(video_keys)

    for parquet_path in parquet_files:
        ep_idx = int(parquet_path.stem.split("_")[-1])
        table = pq.read_table(parquet_path, columns=["task_index"])
        length = table.num_rows
        task_index = int(table.column("task_index")[0].as_py())
        task_name = task_index_to_name.get(task_index, old_episodes.get(ep_idx, {}).get("tasks", ["unknown"])[0])

        chunk = ep_idx // int(info.get("chunks_size", 1000))
        ep_str = f"{ep_idx:06d}"
        for vid_key in video_keys:
            cam = vid_key.removeprefix("observation.images.")
            video_path = (
                root
                / "videos"
                / f"chunk-{chunk:03d}"
                / f"observation.images.{cam}"
                / f"episode_{ep_str}.mp4"
            )
            if not video_path.exists():
                missing_videos.append((ep_idx, cam))

        new_episodes.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_name],
                "length": length,
            }
        )
        total_frames += length

    new_episodes.sort(key=lambda row: row["episode_index"])
    removed = sorted(set(old_episodes) - {row["episode_index"] for row in new_episodes})
    added = sorted(set(row["episode_index"] for row in new_episodes) - set(old_episodes))
    length_changes = [
        ep_idx
        for ep_idx, row in old_episodes.items()
        if ep_idx in {r["episode_index"] for r in new_episodes}
        and old_episodes[ep_idx]["length"] != next(r["length"] for r in new_episodes if r["episode_index"] == ep_idx)
    ]

    print(f"Parquet episodes found: {len(new_episodes)}")
    print(f"Total frames: {total_frames}")
    if removed:
        print(f"Removing from meta (no parquet): {removed}")
    if added:
        print(f"Adding to meta: {added}")
    if length_changes:
        print(f"Length fixes: {length_changes[:20]}{'...' if len(length_changes) > 20 else ''}")
    if missing_videos:
        print(f"Warning: {len(missing_videos)} missing video file(s), e.g. {missing_videos[:5]}")

    if dry_run:
        print("Dry run — meta not written.")
        return

    _write_jsonl(root / "meta" / "episodes.jsonl", new_episodes)

    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        stats_by_ep = {row["episode_index"]: row for row in _read_jsonl(stats_path)}
        _write_jsonl(
            stats_path,
            [stats_by_ep[row["episode_index"]] for row in new_episodes if row["episode_index"] in stats_by_ep],
        )

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    if "total_videos" in info:
        info["total_videos"] = len(new_episodes) * n_cams
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Updated {info_path}")


def _upload_meta_to_hub(*, repo_id: str, hub_repo_id: str | None = None) -> None:
    """Upload only meta/ to Hugging Face (fixes hub listing without re-uploading videos)."""
    from huggingface_hub import HfApi

    root = Path(LEROBOT_HOME / repo_id)
    meta_dir = root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"No meta/ at {meta_dir}")

    target = hub_repo_id or repo_id.replace("local/", "Ishan-Axibo/", 1)
    if not target.startswith("Ishan-Axibo/"):
        target = hub_repo_id or "Ishan-Axibo/piperx_laundry_softfold_ee_merged"

    print(f"Uploading {meta_dir} -> {target}/meta on Hugging Face")
    HfApi().upload_folder(
        repo_id=target,
        folder_path=str(meta_dir),
        path_in_repo="meta",
        repo_type="dataset",
    )
    print("Done. Hub meta now matches your local fixed episode list.")


def main(
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    dry_run: bool = False,
    push_meta: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    sync_dataset_meta(repo_id=repo_id, dry_run=dry_run)
    if push_meta and not dry_run:
        _upload_meta_to_hub(repo_id=repo_id, hub_repo_id=hub_repo_id)


if __name__ == "__main__":
    tyro.cli(main)
