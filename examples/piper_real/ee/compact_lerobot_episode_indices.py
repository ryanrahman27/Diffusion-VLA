"""Close gaps in LeRobot episode_index after deleting an episode.

LeRobot indexes episode_data_index by the episode_index column in parquet.
If episode 153 is removed but later episodes stay numbered 154..269, training
crashes with:
    IndexError: index 269 is out of bounds for dimension 0 with size 269

This renames files and rewrites parquet/meta so indices are contiguous 0..N-1.

Usage:
    uv run python examples/piper_real/ee/compact_lerobot_episode_indices.py \\
        --repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged

    # After fixing locally, re-upload to Hub (example):
    uv run python -c "
from pathlib import Path
from huggingface_hub import HfApi
repo_id = 'Ishan-Axibo/piperx_laundry_softfold_ee_269ep'
root = Path.home() / '.cache/huggingface/lerobot/Ishan-Axibo/piperx_laundry_softfold_ee_merged'
api = HfApi()
api.upload_large_folder(repo_id=repo_id, folder_path=str(root), repo_type='dataset')
api.create_tag(repo_id, tag='v2.1', repo_type='dataset')
"
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _episode_paths(root: Path, info: dict, episode_index: int) -> list[Path]:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    chunk_str = f"{chunk:03d}"
    ep_str = f"{episode_index:06d}"
    paths = [root / "data" / f"chunk-{chunk_str}" / f"episode_{ep_str}.parquet"]
    for key in info.get("features", {}):
        if key.startswith("observation.images."):
            cam = key.removeprefix("observation.images.")
            paths.append(
                root
                / "videos"
                / f"chunk-{chunk_str}"
                / f"observation.images.{cam}"
                / f"episode_{ep_str}.mp4"
            )
    return paths


def _rewrite_parquet_episode_index(path: Path, new_index: int) -> None:
    table = pq.read_table(path)
    if "episode_index" not in table.column_names:
        raise ValueError(f"{path} has no episode_index column")
    col_idx = table.column_names.index("episode_index")
    new_col = pa.array([new_index] * table.num_rows, type=table.schema.field("episode_index").type)
    table = table.set_column(col_idx, "episode_index", new_col)
    pq.write_table(table, path)


def compact_episode_indices(
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

    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    if not episodes:
        raise ValueError("meta/episodes.jsonl is empty")

    indices = sorted(row["episode_index"] for row in episodes)
    expected = list(range(len(indices)))
    if indices == expected:
        print(f"Episode indices already contiguous 0..{len(indices) - 1}. Nothing to do.")
        return

    missing = sorted(set(range(indices[-1] + 1)) - set(indices))
    if len(missing) != 1:
        raise ValueError(
            f"Expected a single gap to close, found missing indices {missing}. "
            "This script handles one deleted episode (one gap) at a time."
        )
    gap = missing[0]
    to_shift = [idx for idx in indices if idx > gap]
    if not to_shift:
        print("No episodes above the gap; nothing to shift.")
        return

    print(f"Closing gap at episode {gap}: renumbering {len(to_shift)} episodes "
          f"({to_shift[0]}..{to_shift[-1]} -> {to_shift[0] - 1}..{to_shift[-1] - 1})")

    # Two-phase rename: old -> tmp -> new (avoids collisions when closing an interior gap).
    tmp_suffix = ".compact_tmp"
    tmp_renames: list[tuple[Path, Path]] = []
    final_renames: list[tuple[Path, Path]] = []

    for old_idx in to_shift:
        new_idx = old_idx - 1
        for old_path, new_path in zip(
            _episode_paths(root, info, old_idx),
            _episode_paths(root, info, new_idx),
            strict=True,
        ):
            tmp_path = old_path.with_name(old_path.name + tmp_suffix)
            tmp_renames.append((old_path, tmp_path))
            final_renames.append((tmp_path, new_path))

    sources_to_clear = {old for old, _ in tmp_renames}

    for old_path, tmp_path in tmp_renames:
        if not old_path.exists():
            raise FileNotFoundError(f"Missing file to rename: {old_path}")
        if tmp_path.exists():
            raise FileExistsError(f"Temporary path already exists: {tmp_path}")
        action = "would rename" if dry_run else "renaming"
        print(f"  {action}: {old_path.name} -> {tmp_path.name}")

    for tmp_path, new_path in final_renames:
        if new_path.exists() and new_path not in sources_to_clear:
            raise FileExistsError(f"Refusing to overwrite existing file: {new_path}")
        action = "would rename" if dry_run else "renaming"
        print(f"  {action}: {tmp_path.name} -> {new_path.name}")

    if dry_run:
        print("Dry run — no files changed.")
        return

    for old_path, tmp_path in tmp_renames:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(tmp_path)

    for tmp_path, new_path in final_renames:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.rename(new_path)

    for old_idx in to_shift:
        new_idx = old_idx - 1
        parquet_path = _episode_paths(root, info, new_idx)[0]
        _rewrite_parquet_episode_index(parquet_path, new_idx)

    new_episodes = []
    for row in episodes:
        ep_idx = int(row["episode_index"])
        if ep_idx > gap:
            row = {**row, "episode_index": ep_idx - 1}
        new_episodes.append(row)
    new_episodes.sort(key=lambda row: row["episode_index"])
    _write_jsonl(root / "meta" / "episodes.jsonl", new_episodes)

    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        new_stats = []
        for row in _read_jsonl(stats_path):
            ep_idx = int(row["episode_index"])
            if ep_idx > gap:
                row = {**row, "episode_index": ep_idx - 1}
            new_stats.append(row)
        new_stats.sort(key=lambda row: row["episode_index"])
        _write_jsonl(stats_path, new_stats)

    info["total_episodes"] = len(new_episodes)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    final_indices = [row["episode_index"] for row in new_episodes]
    print(f"Done. Episodes: {len(new_episodes)}, index range {min(final_indices)}..{max(final_indices)}")


def main(
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    dry_run: bool = False,
) -> None:
    compact_episode_indices(repo_id=repo_id, dry_run=dry_run)


if __name__ == "__main__":
    tyro.cli(main)
