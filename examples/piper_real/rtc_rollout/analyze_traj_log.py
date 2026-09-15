#!/usr/bin/env python3
"""Summarize paper RTC JSONL trajectory logs (--traj-log output).

Usage:
    python analyze_traj_log.py /tmp/paper_rtc_traj.jsonl
    python analyze_traj_log.py /tmp/paper_rtc_traj.jsonl --window 40 --top 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_rows(path: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    meta: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("evt") == "meta":
                meta = rec
                continue
            rows.append(rec)
    return meta, rows


def _merge_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    """Merge tick indices: explicit evt=merge plus large queue refills."""
    out: List[int] = []
    prev_q: Optional[int] = None
    for i, r in enumerate(rows):
        if r.get("evt") == "merge":
            out.append(i)
            prev_q = r.get("q")
            continue
        q = r.get("q")
        if q is not None and prev_q is not None and q - prev_q >= 15:
            out.append(i)
        if q is not None:
            prev_q = q
    # De-duplicate merges within 30 ticks (~250 ms @ 120 Hz).
    deduped: List[int] = []
    for idx in out:
        if deduped and idx - deduped[-1] < 30:
            continue
        deduped.append(idx)
    return deduped


def _per_dim_delta(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [abs(x - y) for x, y in zip(a, b)]


def _window_stats(rows: Sequence[Dict[str, Any]], center: int, half: int) -> Dict[str, Any]:
    lo = max(0, center - half)
    hi = min(len(rows), center + half + 1)
    win = rows[lo:hi]
    d_pub = [r["d_pub"] for r in win if r.get("d_pub") is not None]
    d_raw = [r["d_raw"] for r in win if r.get("d_raw") is not None]
    xf = [r["xf"] for r in win if r.get("xf") is not None]

    pub_jump_dim: Optional[List[float]] = None
    raw_jump_dim: Optional[List[float]] = None
    center_row = rows[center]
    if center > 0:
        prev = rows[center - 1]
        if center_row.get("pub") and prev.get("pub"):
            pub_jump_dim = _per_dim_delta(center_row["pub"], prev["pub"])
        if center_row.get("raw") and prev.get("pub"):
            raw_jump_dim = _per_dim_delta(center_row["raw"], prev["pub"])

    return {
        "center": center,
        "mc": center_row.get("mc"),
        "q": center_row.get("q"),
        "t": center_row.get("t"),
        "max_d_pub": max(d_pub) if d_pub else None,
        "max_d_raw": max(d_raw) if d_raw else None,
        "mean_d_pub": sum(d_pub) / len(d_pub) if d_pub else None,
        "xf_ticks": len(xf),
        "xf_max": max(xf) if xf else None,
        "pub_jump_dim": pub_jump_dim,
        "raw_jump_dim": raw_jump_dim,
        "infer_ms": center_row.get("infer_ms"),
        "merge_d": center_row.get("merge_d"),
    }


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def _joint_labels(n: int) -> List[str]:
    if n == 14:
        return [f"L{j}" for j in range(6)] + ["Lg"] + [f"R{j}" for j in range(6)] + ["Rg"]
    return [f"j{i}" for i in range(n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="JSONL from --traj-log")
    p.add_argument("--window", type=int, default=30, help="±ticks around each merge")
    p.add_argument("--top", type=int, default=12, help="Show worst merges by max d_pub")
    p.add_argument("--d-pub-threshold", type=float, default=0.04, help="Flag jerk if max d_pub exceeds")
    args = p.parse_args()

    meta, rows = _load_rows(args.log)
    if not rows:
        print("No trajectory rows found.", file=sys.stderr)
        return 1

    merges = _merge_indices(rows)
    d_pub_all = [r["d_pub"] for r in rows if r.get("d_pub") is not None]
    d_raw_all = [r["d_raw"] for r in rows if r.get("d_raw") is not None]

    print(f"log: {args.log}")
    if meta:
        print(f"format: {meta.get('format', '?')}")
    print(f"ticks: {len(rows)}  merges: {len(merges)}")
    if d_pub_all:
        print(
            f"d_pub  p50={_fmt(sorted(d_pub_all)[len(d_pub_all) // 2])}  "
            f"p95={_fmt(sorted(d_pub_all)[int(len(d_pub_all) * 0.95)])}  "
            f"max={_fmt(max(d_pub_all))}"
        )
    if d_raw_all:
        print(
            f"d_raw  p50={_fmt(sorted(d_raw_all)[len(d_raw_all) // 2])}  "
            f"p95={_fmt(sorted(d_raw_all)[int(len(d_raw_all) * 0.95)])}  "
            f"max={_fmt(max(d_raw_all))}"
        )

    merge_stats = [_window_stats(rows, i, args.window) for i in merges]
    merge_stats.sort(key=lambda s: (s["max_d_pub"] or 0.0), reverse=True)

    bad = [s for s in merge_stats if (s["max_d_pub"] or 0.0) > args.d_pub_threshold]
    print(f"\nmerges with max d_pub > {args.d_pub_threshold} in ±{args.window} window: {len(bad)}/{len(merges)}")

    print(f"\n{'mc':>4} {'q':>4} {'t':>4}  {'max_pub':>8} {'max_raw':>8}  {'xf#':>4} {'xf_max':>6}  {'merge_d':>7}  note")
    print("-" * 72)
    for s in merge_stats[: args.top]:
        note = ""
        if (s["max_d_pub"] or 0) > args.d_pub_threshold:
            note = "JERK"
        elif (s["max_d_raw"] or 0) > 0.08 and (s["xf_ticks"] or 0) < 5:
            note = "raw jump, weak xfade"
        elif (s["xf_max"] or 0) < 0.5:
            note = "xfade incomplete?"
        print(
            f"{s['mc']!s:>4} {s['q']!s:>4} {s['t']!s:>4}  "
            f"{_fmt(s['max_d_pub']):>8} {_fmt(s['max_d_raw']):>8}  "
            f"{s['xf_ticks']!s:>4} {_fmt(s['xf_max'], 2):>6}  "
            f"{s['merge_d']!s:>7}  {note}"
        )

    # Per-dimension breakdown on worst merge.
    if merge_stats and merge_stats[0].get("pub_jump_dim"):
        worst = merge_stats[0]
        dims = worst["pub_jump_dim"]
        labels = _joint_labels(len(dims))
        ranked = sorted(zip(labels, dims), key=lambda x: x[1], reverse=True)
        print(f"\nWorst merge (mc={worst['mc']}) per-joint |Δpub| at merge tick:")
        print("  " + "  ".join(f"{lab}:{val:.4f}" for lab, val in ranked[:8]))

    if merge_stats and merge_stats[0].get("raw_jump_dim"):
        worst = merge_stats[0]
        dims = worst["raw_jump_dim"]
        labels = _joint_labels(len(dims))
        ranked = sorted(zip(labels, dims), key=lambda x: x[1], reverse=True)
        print(f"Same merge |Δraw| vs previous pub (pre-crossfade discontinuity):")
        print("  " + "  ".join(f"{lab}:{val:.4f}" for lab, val in ranked[:8]))

    # Crossfade effectiveness: d_pub on ticks with xf>0 vs xf is None.
    xf_pub: List[float] = []
    no_xf_pub: List[float] = []
    for r in rows:
        if r.get("d_pub") is None:
            continue
        if r.get("xf") is not None:
            xf_pub.append(r["d_pub"])
        else:
            no_xf_pub.append(r["d_pub"])
    if xf_pub:
        print(
            f"\ncrossfade ticks: {len(xf_pub)}  mean d_pub={sum(xf_pub)/len(xf_pub):.4f}  "
            f"max={max(xf_pub):.4f}"
        )
    if no_xf_pub:
        print(
            f"normal ticks:    {len(no_xf_pub)}  mean d_pub={sum(no_xf_pub)/len(no_xf_pub):.4f}  "
            f"max={max(no_xf_pub):.4f}"
        )

    # Guidance overlap hint from logged threshold.
    thr = rows[0].get("thr")
    if thr is not None:
        print(
            f"\nRTC infer threshold (s_min) in log: {thr}. "
            "If s_min ≈ H, guidance overlap is ~0 and merge raw jumps stay large."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
