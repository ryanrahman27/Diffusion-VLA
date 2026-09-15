"""Inspect a LeRobot depth PNG (raw Z16 I;16) and write a visible preview."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tyro

DEFAULT_DEPTH_PNG = (
    Path.home()
    / ".cache/huggingface/lerobot/Ishan-Axibo/piperx_flatten_depth/images"
    / "observation.depth.cam_right_wrist/episode_000003/frame_000200.png"
)

CmapName = Literal["turbo", "viridis", "plasma", "inferno", "magma", "jet"]


def depth_to_colormap(
    depth_mm: np.ndarray,
    *,
    vmin_mm: float | None = None,
    vmax_mm: float | None = None,
    cmap: str = "turbo",
) -> np.ndarray:
    """Map valid Z16 depth (mm) to RGB; invalid/zero pixels stay black."""
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)

    vmin = float(depth_mm[valid].min() if vmin_mm is None else vmin_mm)
    vmax = float(depth_mm[valid].max() if vmax_mm is None else vmax_mm)
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.zeros(depth_mm.shape, dtype=np.float32)
    norm[valid] = (depth_mm[valid].astype(np.float32) - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    colored = (plt.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored


def main(
    png: Path = DEFAULT_DEPTH_PNG,
    out: Path = Path("depth_viz.png"),
    depth_scale_m: float = 0.001,
    cmap: CmapName = "turbo",
    max_depth_m: float | None = None,
    min_depth_m: float | None = None,
    grayscale: bool = False,
) -> None:
    """Load a depth PNG, print stats, and save a colormap visualization."""
    png = png.expanduser().resolve()
    if not png.exists():
        raise FileNotFoundError(
            f"Depth PNG not found: {png}\n"
            "Pass --png to a frame under your dataset, e.g.\n"
            "  ~/.cache/huggingface/lerobot/<repo>/images/"
            "observation.depth.cam_front/episode_000000/frame_000000.png"
        )

    im = np.array(Image.open(png))
    valid = im[im > 0]
    print(f"file: {png}")
    print(f"shape: {im.shape}  dtype: {im.dtype}")
    print(f"min: {im.min()}  max: {im.max()}  mean: {im.mean():.1f}  nonzero: {np.count_nonzero(im)}")
    if valid.size:
        print(
            f"valid depth (mm): min={valid.min()} max={valid.max()} median={np.median(valid):.0f}  "
            f"meters: {valid.min() * depth_scale_m:.3f}–{valid.max() * depth_scale_m:.3f}"
        )
    else:
        print("WARNING: all pixels are zero — no depth recorded")

    vmin_mm = None if min_depth_m is None else min_depth_m / depth_scale_m
    vmax_mm = None if max_depth_m is None else max_depth_m / depth_scale_m
    if vmin_mm is not None or vmax_mm is not None:
        print(f"colormap range (m): {min_depth_m} – {max_depth_m}")

    out = out.expanduser().resolve()
    if grayscale:
        viz = (im.astype(np.float32) / im.max() * 255).astype(np.uint8) if im.max() else im.astype(np.uint8)
        Image.fromarray(viz).save(out)
    else:
        rgb = depth_to_colormap(im, vmin_mm=vmin_mm, vmax_mm=vmax_mm, cmap=cmap)
        Image.fromarray(rgb).save(out)

    print(f"wrote preview ({cmap if not grayscale else 'grayscale'}): {out}")
    print("  close = dark / blue-ish, far = warm / yellow-red (turbo); black = no depth")


if __name__ == "__main__":
    tyro.cli(main)
