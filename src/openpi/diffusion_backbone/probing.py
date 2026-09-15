"""Representation probing for the diffusion-backbone study (paper Sec 9.2).

Fits capacity-matched probes on the frozen action-token features dumped by
scripts/extract_probe_features.py and compares C (Dream, diffusion) vs D (Qwen, AR).
Because C and D share hidden width (3584), the SAME probe is capacity-matched across
them with no PCA step (PCA-to-common-dim is only needed for the optional stock-Gemma
comparison at width 2048).

Targets:
  - action_next  : immediate action (real DoF)             regression, R^2
  - action_chunk : full H-step action chunk (real DoF)     regression, R^2
  - state        : proprioceptive state                    regression, R^2
  - gripper      : gripper command, median-split           classification, acc + AUROC

Probes: linear (Ridge / LogisticRegression) is the headline "linear decodability"
number; a small MLP is the nonlinear ceiling. Each is fit over N seeds (different
train/test splits) for mean +/- std.

CLI:
    python -m openpi.diffusion_backbone.probing \
        --c probe_feats/c_seed0.npz --d probe_feats/d_seed0.npz \
        --out src/openpi/diffusion_backbone/docs/probing_results.md
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

REAL_ACTION_DIM = 7  # LIBERO Franka: 6 EEF delta + 1 gripper (action_dim padded to 32)
GRIPPER_DIM = 6  # index of the gripper command within the real DoF


@dataclasses.dataclass
class Target:
    name: str
    kind: str  # "reg" or "clf"


TARGETS = [
    Target("action_next", "reg"),
    Target("action_chunk", "reg"),
    Target("state", "reg"),
    Target("gripper", "clf"),
]


def _build_target(dump: dict, name: str) -> np.ndarray:
    actions = dump["actions"]  # [N, H, A]
    if name == "action_next":
        y = actions[:, 0, :REAL_ACTION_DIM]
    elif name == "action_chunk":
        y = actions[:, :, :REAL_ACTION_DIM].reshape(actions.shape[0], -1)
    elif name == "state":
        y = dump["state"]
    elif name == "gripper":
        g = actions[:, 0, GRIPPER_DIM]
        y = (g > np.median(g)).astype(np.int64)  # balanced binary
    else:
        raise ValueError(name)
    return y


def _drop_constant(y: np.ndarray) -> np.ndarray:
    """Drop zero-variance columns (padding dims) so R^2 isn't polluted by constants."""
    if y.ndim == 1:
        return y
    keep = y.std(axis=0) > 1e-6
    return y[:, keep] if keep.any() else y


def _fit_once(X, y, kind, probe, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    if kind == "reg":
        if probe == "linear":
            m = Ridge(alpha=10.0)
        else:
            m = MLPRegressor(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=seed)
        m.fit(Xtr, ytr)
        return float(r2_score(yte, m.predict(Xte), multioutput="uniform_average"))

    # classification -> return (accuracy, auroc)
    if probe == "linear":
        m = LogisticRegression(max_iter=1000, C=1.0)
    else:
        m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=seed)
    m.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, m.predict(Xte)))
    try:
        auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    except (ValueError, IndexError):
        auc = float("nan")
    return acc, auc


def _agg(vals):
    a = np.array(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def run(c_path: str, d_path: str, out_md: str | None, n_seeds: int = 5, feature: str = "feat_mean") -> str:
    c = dict(np.load(c_path, allow_pickle=True))
    d = dict(np.load(d_path, allow_pickle=True))

    # Fairness check: C and D must be probed on the SAME frames.
    n = min(len(c["actions"]), len(d["actions"]))
    aligned = np.allclose(c["actions"][:n], d["actions"][:n], atol=1e-4)
    align_note = "OK (identical frames)" if aligned else "WARNING: C/D frames differ -- comparison not matched!"

    Xc, Xd = c[feature][:n], d[feature][:n]
    lines = [
        "# Representation probing: C (Dream, diffusion) vs D (Qwen, AR)",
        "",
        f"Feature: `{feature}` (action-token hidden state, tau=1). Samples: {n}. "
        f"Seeds: {n_seeds}. Sample alignment: {align_note}",
        "",
        "R^2 for regression targets; accuracy (AUROC) for gripper. Higher is better; "
        "**C-D > 0 means the diffusion prior is more decodable.**",
        "",
        "| Target | Probe | C | D | C - D |",
        "|--------|-------|--:|--:|--:|",
    ]

    for t in TARGETS:
        yc, yd = _build_target(c, t.name), _build_target(d, t.name)
        yc, yd = yc[:n], yd[:n]
        if t.kind == "reg":
            yc, yd = _drop_constant(yc), _drop_constant(yd)
        for probe in ("linear", "mlp"):
            if t.kind == "reg":
                cv = [_fit_once(Xc, yc, "reg", probe, s) for s in range(n_seeds)]
                dv = [_fit_once(Xd, yd, "reg", probe, s) for s in range(n_seeds)]
                cm, cs = _agg(cv)
                dm, ds = _agg(dv)
                lines.append(
                    f"| {t.name} | {probe} | {cm:.3f} ± {cs:.3f} | {dm:.3f} ± {ds:.3f} | {cm - dm:+.3f} |"
                )
            else:
                cv = [_fit_once(Xc, yc, "clf", probe, s) for s in range(n_seeds)]
                dv = [_fit_once(Xd, yd, "clf", probe, s) for s in range(n_seeds)]
                cacc, _ = _agg([a for a, _ in cv])
                cauc, _ = _agg([u for _, u in cv])
                dacc, _ = _agg([a for a, _ in dv])
                dauc, _ = _agg([u for _, u in dv])
                lines.append(
                    f"| {t.name} | {probe} | {cacc:.3f} ({cauc:.3f}) | {dacc:.3f} ({dauc:.3f}) | {cacc - dacc:+.3f} |"
                )

    report = "\n".join(lines) + "\n"
    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(report)
    print(report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c", required=True, help="C features .npz")
    p.add_argument("--d", required=True, help="D features .npz")
    p.add_argument("--out", default=None, help="markdown output path")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--feature", default="feat_mean", choices=["feat_mean", "feat_first"])
    args = p.parse_args()
    run(args.c, args.d, args.out, args.n_seeds, args.feature)


if __name__ == "__main__":
    main()
