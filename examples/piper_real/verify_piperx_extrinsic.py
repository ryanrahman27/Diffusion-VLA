"""
Verify PiperX bimanual base extrinsic and rot6d conventions before training with rel EE.

What this checks
----------------
1. **Gram-Schmidt rot6d** round-trip matches ``episode_recorder.rpy_to_rot6d`` layout.
2. **Live / logged EE poses** from the Piper SDK (via teleop JSON or HDF5 ``eef_6d``).
3. **Implied base offset** ``T_L @ inv(T_R)`` when both grippers share the same world pose.
   Rotation should be ~identity (parallel mounts); translation should match
   ``PIPERX_RIGHT_BASE_IN_LEFT_BASE``.

Procedure (live)
----------------
1. Start teleop publisher so port 3335 streams follower EE poses.
2. Command both arms to the **same physical pose** (grippers coincident / touching).
3. Hold steady, then run:

    uv run python examples/piper_real/verify_piperx_extrinsic.py \\
        --teleop-addr tcp://127.0.0.1:3335

Offline (HDF5 frame):

    uv run python examples/piper_real/verify_piperx_extrinsic.py \\
        --hdf5 /home/axibo/piperx_lerobot_setup/episodes/laundry/episode_000000.hdf5 \\
        --frame 100

Rot6d self-test only:

    uv run python examples/piper_real/verify_piperx_extrinsic.py --check-rot6d-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from openpi.policies.piperx_rel_pose import PIPERX_RIGHT_BASE_IN_LEFT_BASE
from openpi.policies.piperx_rel_pose import base_right_to_base_left
from openpi.policies.piperx_rel_pose import compute_inter_gripper_rel
from openpi.policies.piperx_rel_pose import implied_base_right_in_left
from openpi.policies.piperx_rel_pose import pose6d_to_se3
from openpi.policies.piperx_rel_pose import relative_pose_9d_from_transform
from openpi.policies.piperx_rel_pose import se3_inverse
from openpi.policies.piperx_rel_pose import rotation_angle_rad
from openpi.policies.piperx_rel_pose import rot6d_to_rotation_matrix
from openpi.policies.piperx_rel_pose import rpy_to_rot6d
from openpi.policies.piperx_rel_pose import verify_rot6d_roundtrip


def _episode_recorder_rpy_to_rot6d_reference(rpy: list[float]) -> np.ndarray:
    """Byte-for-byte reference of ``episode_recorder.rpy_to_rot6d`` (no heavy imports)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.asarray(
        [
            cy * cp,
            sy * cp,
            -sp,
            cy * sp * sr - sy * cr,
            sy * sp * sr + cy * cr,
            cp * sr,
        ],
        dtype=np.float32,
    )


def _fmt_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.5f}" for v in values) + "]"


def check_rot6d_conventions() -> bool:
    """Confirm Gram-Schmidt matches recorder byte layout on representative RPY samples."""
    print("== rot6d / Gram-Schmidt convention ==")
    samples = [
        ("zero", [0.0, 0.0, 0.0]),
        ("roll", [0.35, 0.0, 0.0]),
        ("pitch", [0.0, -0.25, 0.0]),
        ("yaw", [0.0, 0.0, 0.8]),
        ("fold-like", [1.07, 0.14, -1.67]),
    ]
    ok = True
    recorder_ok = True
    for name, rpy in samples:
        passed, roundtrip_err = verify_rot6d_roundtrip(rpy)
        rot6d = rpy_to_rot6d(rpy)
        recorder_rot6d = _episode_recorder_rpy_to_rot6d_reference(list(rpy))
        recorder_match = np.allclose(rot6d, recorder_rot6d, atol=1e-6)
        print(
            f"  {name:10s}  rpy={_fmt_vec(np.asarray(rpy))}  "
            f"rot6d={_fmt_vec(rot6d)}  roundtrip_err={roundtrip_err:.2e}  "
            f"recorder_match={recorder_match}  "
            f"{'OK' if passed and recorder_match else 'FAIL'}"
        )
        ok = ok and passed and recorder_match
        recorder_ok = recorder_ok and recorder_match

    # Byte layout: first two *columns* of R flattened (r00,r10,r20,r01,r11,r21).
    rpy = [0.1, -0.2, 0.3]
    rot6d = rpy_to_rot6d(rpy)
    recovered = rot6d_to_rotation_matrix(rot6d)
    col0_match = np.allclose(recovered[:, 0], rot6d[:3], atol=1e-5)
    col1_raw = np.array(rot6d[3:6], dtype=np.float64)
    col1_match = np.allclose(recovered[:, 1], col1_raw, atol=1e-3) or np.linalg.norm(col1_raw) < 1e-6
    print(f"  column layout: col0 exact={col0_match}, col1 stored-raw close={col1_match}")
    print(f"  episode_recorder byte match on all samples: {recorder_ok}")
    ok = ok and col0_match
    print()
    return ok


def _pose_from_follower(side: dict) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(side["ee_xyz"][:3], dtype=np.float64)
    rot6d = rpy_to_rot6d(side["ee_rpy"][:3])
    return xyz, rot6d


def _pose_from_eef_6d_slice(values: np.ndarray, *, left: bool) -> tuple[np.ndarray, np.ndarray]:
    start = 0 if left else 10
    xyz = np.asarray(values[start : start + 3], dtype=np.float64)
    rot6d = np.asarray(values[start + 3 : start + 9], dtype=np.float32)
    return xyz, rot6d


def check_rel_encoding_order(eef_6d: np.ndarray) -> bool:
    """Relative 9-D feature must be [xyz(3), rot6d(6)], not [rot6d(6), xyz(3)]."""
    rel = compute_inter_gripper_rel(eef_6d)
    left_xyz, left_rot6d = _pose_from_eef_6d_slice(eef_6d, left=True)
    right_xyz, right_rot6d = _pose_from_eef_6d_slice(eef_6d, left=False)
    left_t = pose6d_to_se3(left_xyz, left_rot6d)
    right_in_left = base_right_to_base_left(PIPERX_RIGHT_BASE_IN_LEFT_BASE) @ pose6d_to_se3(
        right_xyz, right_rot6d
    )
    expected = relative_pose_9d_from_transform(se3_inverse(left_t) @ right_in_left)

    xyz_match = np.allclose(rel[:3], expected[:3], atol=1e-4)
    rot_match = np.allclose(rel[3:], expected[3:], atol=1e-4)
    wrong_order_rot_first = np.allclose(rel[:6], expected[3:], atol=1e-4) and np.allclose(
        rel[6:], expected[:3], atol=1e-4
    )
    print("== relative pose encoding order ==")
    print(f"  rel layout      = xyz {_fmt_vec(rel[:3])}  rot6d {_fmt_vec(rel[3:])}")
    print(f"  expected layout = xyz {_fmt_vec(expected[:3])}  rot6d {_fmt_vec(expected[3:])}")
    print(f"  [xyz, rot6d] match: {xyz_match and rot_match}")
    print(f"  wrong [rot6d, xyz] would match: {wrong_order_rot_first}")
    print()
    return bool(xyz_match and rot_match and not wrong_order_rot_first)


def check_extrinsic(
    left_xyz: np.ndarray,
    left_rot6d: np.ndarray,
    right_xyz: np.ndarray,
    right_rot6d: np.ndarray,
    *,
    expected_translation: np.ndarray,
    rot_tol_deg: float,
    trans_tol_m: float,
) -> bool:
    print("== base extrinsic verification ==")
    print("  Assumes both grippers are at the same world pose (coincident EE frames).")
    print(f"  T_L (left EE in left base):  xyz={_fmt_vec(left_xyz)}")
    print(f"  T_R (right EE in right base): xyz={_fmt_vec(right_xyz)}")

    r_left = rot6d_to_rotation_matrix(left_rot6d)
    r_right = rot6d_to_rotation_matrix(right_rot6d)
    rot_delta_deg = math.degrees(rotation_angle_rad(r_left @ r_right.T))
    print(f"  R_L vs R_R geodesic angle: {rot_delta_deg:.3f} deg")

    implied = implied_base_right_in_left(left_xyz, left_rot6d, right_xyz, right_rot6d)
    implied_rot = implied[:3, :3]
    implied_trans = implied[:3, 3]
    rot_err_deg = math.degrees(rotation_angle_rad(implied_rot))
    trans_err = float(np.linalg.norm(implied_trans - expected_translation))

    print(f"  implied base rotation off I: {rot_err_deg:.3f} deg")
    print(f"  implied base translation:    {_fmt_vec(implied_trans)}")
    print(f"  configured translation:    {_fmt_vec(expected_translation)}")
    print(f"  translation L2 error:        {trans_err:.4f} m")

    rot_ok = rot_err_deg <= rot_tol_deg
    trans_ok = trans_err <= trans_tol_m
    print(f"  rotation within {rot_tol_deg:.1f} deg: {rot_ok}")
    print(f"  translation within {trans_tol_m:.3f} m: {trans_ok}")
    print()
    return rot_ok and trans_ok


def read_teleop_snapshot(addr: str, timeout_s: float) -> dict:
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required for --teleop-addr") from exc

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))

    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            payload = sock.recv()
            msg = json.loads(payload.decode("utf-8"))
            if "follower_left" in msg and "follower_right" in msg:
                return msg
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.05)
    raise TimeoutError(f"No teleop JSON with follower poses on {addr} ({last_err})")


def read_hdf5_frame(path: Path, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        eef_6d = np.asarray(f["/observations/eef_6d"][frame], dtype=np.float32)
    left_xyz, left_rot6d = _pose_from_eef_6d_slice(eef_6d, left=True)
    right_xyz, right_rot6d = _pose_from_eef_6d_slice(eef_6d, left=False)
    return left_xyz, left_rot6d, right_xyz, right_rot6d, eef_6d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teleop-addr", default=None, help="ZMQ SUB address, e.g. tcp://127.0.0.1:3335")
    parser.add_argument("--hdf5", type=Path, default=None, help="HDF5 episode path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for --hdf5")
    parser.add_argument("--teleop-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--expected-base-translation",
        type=float,
        nargs=3,
        default=tuple(PIPERX_RIGHT_BASE_IN_LEFT_BASE.tolist()),
        metavar=("X", "Y", "Z"),
        help="Right base origin in left base frame (default: configured constant)",
    )
    parser.add_argument("--rot-tol-deg", type=float, default=8.0, help="Max |R - I| for implied base rotation")
    parser.add_argument("--trans-tol-m", type=float, default=0.03, help="Max translation error vs configured offset")
    parser.add_argument(
        "--check-rot6d-only",
        action="store_true",
        help="Only run rot6d / Gram-Schmidt checks (no live pose required)",
    )
    parser.add_argument(
        "--require-extrinsic",
        action="store_true",
        help="Fail if base extrinsic check fails (default for --teleop-addr; off for --hdf5)",
    )
    args = parser.parse_args()

    expected_translation = np.asarray(args.expected_base_translation, dtype=np.float64)
    all_ok = True
    all_ok = check_rot6d_conventions() and all_ok

    if args.check_rot6d_only:
        return 0 if all_ok else 1

    if args.teleop_addr is None and args.hdf5 is None:
        print("Provide --teleop-addr or --hdf5 (or use --check-rot6d-only).", file=sys.stderr)
        return 2

    eef_6d = None
    if args.teleop_addr is not None:
        print(f"== reading live teleop from {args.teleop_addr} ==")
        msg = read_teleop_snapshot(args.teleop_addr, args.teleop_timeout_s)
        left_xyz, left_rot6d = _pose_from_follower(msg["follower_left"])
        right_xyz, right_rot6d = _pose_from_follower(msg["follower_right"])
        print("  snapshot acquired.\n")
    else:
        print(f"== reading HDF5 {args.hdf5} frame {args.frame} ==")
        left_xyz, left_rot6d, right_xyz, right_rot6d, eef_6d = read_hdf5_frame(args.hdf5, args.frame)
        print(
            "  NOTE: HDF5 frames are usually *not* coincident-gripper poses; "
            "extrinsic translation check may fail unless you pick a calibration frame.\n"
        )

    if eef_6d is None:
        eef_6d = np.concatenate(
            [
                left_xyz.astype(np.float32),
                left_rot6d.astype(np.float32),
                np.zeros(1, dtype=np.float32),
                right_xyz.astype(np.float32),
                right_rot6d.astype(np.float32),
                np.zeros(1, dtype=np.float32),
            ]
        )

    all_ok = check_rel_encoding_order(eef_6d) and all_ok

    run_extrinsic = args.require_extrinsic or args.teleop_addr is not None
    if run_extrinsic:
        all_ok = (
            check_extrinsic(
                left_xyz,
                left_rot6d,
                right_xyz,
                right_rot6d,
                expected_translation=expected_translation,
                rot_tol_deg=args.rot_tol_deg,
                trans_tol_m=args.trans_tol_m,
            )
            and all_ok
        )
    else:
        print("== base extrinsic verification ==")
        print("  SKIPPED (use --teleop-addr with coincident grippers or --require-extrinsic).")
        print()

    if all_ok:
        parts = ["rot6d convention", "rel encoding order"]
        if run_extrinsic:
            parts.append("extrinsic")
        print(f"PASS: {', '.join(parts)} checks succeeded.")
        return 0
    print("FAIL: one or more checks did not pass.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
