"""Remove one episode from a local LeRobot dataset (no re-convert / no Hub push).

Fixes corrupt or unwanted episodes by deleting parquet + videos and updating meta.

Usage (on your VM):
    uv run python examples/piper_real/ee/remove_lerobot_episode.py \\
        --repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged \\
        --episode 153
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
import tyro


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def remove_episode(
    *,
    repo_id: str,
    episode_index: int,
    dry_run: bool = False,
) -> None:
    root = Path(LEROBOT_HOME / repo_id)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = _read_jsonl(episodes_path)
    match = [ep for ep in episodes if ep["episode_index"] == episode_index]
    if not match:
        raise ValueError(f"Episode {episode_index} not in {episodes_path}")
    ep_meta = match[0]
    length = int(ep_meta["length"])

    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    chunk = episode_index // int(info.get("chunks_size", 1000))
    chunk_str = f"{chunk:03d}"
    ep_str = f"{episode_index:06d}"

    paths_to_delete: list[Path] = [
        root / "data" / f"chunk-{chunk_str}" / f"episode_{ep_str}.parquet",
    ]
    for key in info.get("features", {}):
        if key.startswith("observation.images."):
            cam = key.removeprefix("observation.images.")
            paths_to_delete.append(
                root
                / "videos"
                / f"chunk-{chunk_str}"
                / f"observation.images.{cam}"
                / f"episode_{ep_str}.mp4"
            )

    print(f"Removing episode {episode_index} ({length} frames, task={ep_meta.get('tasks')})")
    for path in paths_to_delete:
        status = "would delete" if dry_run else "deleting"
        print(f"  {status}: {path}" if path.exists() else f"  missing (skip): {path}")

    if dry_run:
        print("Dry run only — no files changed.")
        return

    for path in paths_to_delete:
        path.unlink(missing_ok=True)

    _write_jsonl(
        episodes_path,
        [ep for ep in episodes if ep["episode_index"] != episode_index],
    )

    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        stats = _read_jsonl(stats_path)
        _write_jsonl(
            stats_path,
            [row for row in stats if row.get("episode_index") != episode_index],
        )

    info["total_episodes"] = int(info.get("total_episodes", 0)) - 1
    info["total_frames"] = int(info.get("total_frames", 0)) - length
    if "total_videos" in info:
        n_cams = sum(1 for k in info["features"] if k.startswith("observation.images."))
        info["total_videos"] = int(info["total_videos"]) - n_cams
    n_eps = info["total_episodes"]
    info["splits"] = {"train": f"0:{n_eps}"}

    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Updated {info_path}: {info['total_episodes']} episodes, {info['total_frames']} frames")


def main(
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    episode: int = 153,
    dry_run: bool = False,
) -> None:
    remove_episode(repo_id=repo_id, episode_index=episode, dry_run=dry_run)


if __name__ == "__main__":
    tyro.cli(main)
