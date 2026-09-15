"""Delete LeRobot episode files on Hugging Face that are not in local meta.

upload_large_folder does not remove stale remote files. After compacting episode
indices (269 -> 268), episode_000269.parquet may remain on the Hub and break
training (503161 frames loaded vs 495816 in meta).

Usage:
    uv run python examples/piper_real/ee/delete_hub_orphan_episodes.py \\
        --repo-id Ishan-Axibo/piperx_laundry_softfold_ee_merged \\
        --hub-repo-id Ishan-Axibo/piperx_laundry_softfold_ee_269ep
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
import tyro

_EP_RE = re.compile(r"episode_(\d+)\.(parquet|mp4)$")


def delete_hub_orphans(
    *,
    repo_id: str,
    hub_repo_id: str,
    dry_run: bool = False,
) -> None:
    root = Path(LEROBOT_HOME / repo_id)
    episodes_path = root / "meta" / "episodes.jsonl"
    valid = {
        int(json.loads(line)["episode_index"])
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    }
    print(f"Local meta: {len(valid)} episodes, max index {max(valid)}")

    api = HfApi()
    remote_files = list_repo_files(hub_repo_id, repo_type="dataset")
    orphans: list[str] = []
    for path in remote_files:
        match = _EP_RE.search(path)
        if match and int(match.group(1)) not in valid:
            orphans.append(path)

    if not orphans:
        print("No orphan episode files on Hub.")
        return

    print(f"Found {len(orphans)} orphan file(s) on Hub:")
    for path in sorted(orphans):
        print(f"  {'would delete' if dry_run else 'deleting'}: {path}")

    if dry_run:
        return

    for path in orphans:
        api.delete_file(path, repo_id=hub_repo_id, repo_type="dataset")
    print("Done.")


def main(
    repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_merged",
    hub_repo_id: str = "Ishan-Axibo/piperx_laundry_softfold_ee_269ep",
    dry_run: bool = False,
) -> None:
    delete_hub_orphans(repo_id=repo_id, hub_repo_id=hub_repo_id, dry_run=dry_run)


if __name__ == "__main__":
    tyro.cli(main)
