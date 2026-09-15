"""Delete local LeRobot episode files not listed in meta/episodes.jsonl.

LeRobot loads ALL parquet files under data/ (load_dataset data_dir=...). Stale
episode_000269.parquet after compacting causes 270/270 files and timestamp errors
even when Hub meta is correct.

Usage:
    uv run python examples/piper_real/ee/delete_local_orphan_episodes.py \\
        --repo-id Ishan-Axibo/piperx_laundry_softfold_ee_269ep
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
import tyro

_EP_RE = re.compile(r"episode_(\d+)\.(parquet|mp4)$")


def delete_local_orphans(
    *,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    root = Path(LEROBOT_HOME / repo_id)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    episodes_path = root / "meta" / "episodes.jsonl"
    valid = {
        int(json.loads(line)["episode_index"])
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    }
    print(f"Meta: {len(valid)} episodes, max index {max(valid)}")

    orphans: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = _EP_RE.search(path.name)
        if match and int(match.group(1)) not in valid:
            orphans.append(path)

    if not orphans:
        print("No orphan episode files locally.")
        return

    print(f"Found {len(orphans)} orphan file(s):")
    for path in sorted(orphans):
        print(f"  {'would delete' if dry_run else 'deleting'}: {path.relative_to(root)}")

    if dry_run:
        return

    for path in orphans:
        path.unlink()
    print("Done.")


def main(
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_269ep",
    dry_run: bool = False,
) -> None:
    delete_local_orphans(repo_id=repo_id, dry_run=dry_run)


if __name__ == "__main__":
    tyro.cli(main)
