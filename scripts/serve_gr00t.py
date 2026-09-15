"""Serve a GR00T N1.7 PiperX checkpoint behind the openpi websocket policy protocol.

Same client contract as serve_policy.py / serve_molmoact2.py — the PiperX robot
runtime (openpi-client WebsocketClientPolicy + ActionChunkBroker) is unchanged:

  - client sends obs:  {"state": (14,) float raw robot units,
                        "images": {"cam_front"|"cam_left_wrist"|"cam_right_wrist":
                                   (3, H, W) uint8 CHW},
                        "prompt": str (optional)}
  - server returns:    {"actions": (16, 14) float32 raw robot units,
                        "policy_timing": {...}}

GR00T specifics:
  - Action horizon is 16 (vs 25 for MolmoAct2) -> robot client must use
    ActionChunkBroker(action_horizon=16), or read it from server metadata.
  - The 14-D layout matches the dataset: left_joint_1..6, left_gripper,
    right_joint_1..6, right_gripper (joints rad, grippers meters 0-0.07).
  - Arm actions were trained RELATIVE; Gr00tPolicy.decode_action converts them
    back to ABSOLUTE joint targets using the current state, so the wire carries
    absolute targets exactly like the pi0.5/MolmoAct2 paths.
  - The PiperX modality config (scripts/piperx_gr00t_config.py) must be
    registered before loading; this script imports it automatically.

Run inside the Isaac-GR00T uv env (python 3.10), plus
`uv pip install -e <piperx-openpi>/packages/openpi-client websockets`:

  python scripts/serve_gr00t.py \
    --checkpoint axiboai/gr00t-n1.7-piperx-flatten \
    --port 8000
"""

import argparse
import dataclasses
import importlib.util
import logging
import pathlib
import sys
import time

import numpy as np

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)

EXPECTED_CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist")
ACTION_DIM = 14
ACTION_HORIZON = 16
# (key, state slice) in dataset order; must match meta/modality.json.
STATE_SLICES = (
    ("left_arm", slice(0, 6)),
    ("left_gripper", slice(6, 7)),
    ("right_arm", slice(7, 13)),
    ("right_gripper", slice(13, 14)),
)


def _import_module_from_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_websocket_policy_server():
    """Load openpi's WebsocketPolicyServer without importing the openpi (JAX) package."""
    try:
        from openpi.serving import websocket_policy_server  # noqa: PLC0415

        return websocket_policy_server
    except ImportError:
        here = pathlib.Path(__file__).resolve().parent
        candidates = [
            here.parent / "src/openpi/serving/websocket_policy_server.py",  # repo checkout
            here / "websocket_policy_server.py",  # standalone copy
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "websocket_policy_server.py not found; run from a piperx-openpi checkout "
                "or place a copy next to this script."
            )
        return _import_module_from_path("websocket_policy_server", path)


class Gr00tServerPolicy(_base_policy.BasePolicy):
    """Adapts a GR00T N1.7 checkpoint to the openpi BasePolicy interface."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str = "cuda:0",
        default_prompt: str = "",
        modality_config: str | None = None,
        revision: str | None = None,
    ) -> None:
        self._default_prompt = default_prompt

        # Register the PiperX NEW_EMBODIMENT modality config before loading the policy.
        config_path = pathlib.Path(
            modality_config or pathlib.Path(__file__).resolve().parent / "piperx_gr00t_config.py"
        )
        _import_module_from_path("piperx_gr00t_config", config_path)
        logger.info("Registered modality config from %s", config_path)

        local = pathlib.Path(checkpoint_dir).expanduser()
        if local.is_dir():
            checkpoint_dir = str(local.resolve())
        elif revision is not None:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            logger.info("Downloading %s @ %s", checkpoint_dir, revision)
            checkpoint_dir = snapshot_download(checkpoint_dir, revision=revision)

        from gr00t.policy import Gr00tPolicy  # noqa: PLC0415

        logger.info("Loading GR00T checkpoint from %s", checkpoint_dir)
        self._policy = Gr00tPolicy(
            embodiment_tag="NEW_EMBODIMENT",
            model_path=checkpoint_dir,
            device=device,
            strict=False,  # we validate shapes ourselves; strict mode slows serving
        )
        self._checkpoint = checkpoint_dir

    @property
    def metadata(self) -> dict:
        return {
            "policy": "gr00t-n1.7",
            "checkpoint": self._checkpoint,
            "n_action_steps": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
        }

    def _build_obs(self, obs: dict) -> dict:
        images = obs["images"]
        video = {}
        for cam in EXPECTED_CAMERAS:
            if cam not in images:
                raise ValueError(f"Observation missing camera {cam!r}; got {sorted(images)}")
            arr = np.asarray(images[cam])
            if arr.dtype != np.uint8 or arr.ndim != 3:
                raise ValueError(f"{cam}: expected uint8 3D image, got {arr.dtype} {arr.shape}")
            if arr.shape[0] == 3:  # CHW (PiperX runtime) -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] != 3:
                raise ValueError(f"{cam}: cannot interpret shape {arr.shape} as RGB")
            video[cam] = np.ascontiguousarray(arr)[None, None]  # (B=1, T=1, H, W, 3)

        state14 = np.asarray(obs["state"], dtype=np.float32)
        if state14.shape != (ACTION_DIM,):
            raise ValueError(f"state: expected shape ({ACTION_DIM},), got {state14.shape}")
        state = {key: state14[sl][None, None] for key, sl in STATE_SLICES}  # (1, 1, d)

        prompt = obs.get("prompt") or self._default_prompt
        return {
            "video": video,
            "state": state,
            "language": {"annotation.human.task_description": [[prompt]]},
        }

    def infer(self, obs: dict, *, noise=None) -> dict:  # noqa: ARG002 (noise is pi0-specific)
        start = time.monotonic()
        action, _info = self._policy.get_action(self._build_obs(obs))
        # Reassemble (16, 14) in dataset order from per-group (B=1, 16, d) arrays.
        chunk = np.concatenate([np.asarray(action[key])[0] for key, _ in STATE_SLICES], axis=-1)
        chunk = chunk.astype(np.float32)
        if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"Unexpected action chunk shape {chunk.shape}")
        return {
            "actions": chunk,
            "policy_timing": {"infer_ms": (time.monotonic() - start) * 1000},
        }

    def reset(self) -> None:
        pass


@dataclasses.dataclass
class Args:
    checkpoint: str
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"
    default_prompt: str = "pick towel from pile, fold and stack"
    modality_config: str | None = None
    revision: str | None = None
    no_warmup: bool = False


def _parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Local checkpoint-<step> dir or HF repo id")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--default-prompt", default="pick towel from pile, fold and stack")
    p.add_argument("--modality-config", default=None, help="Path to piperx_gr00t_config.py override")
    p.add_argument("--revision", default=None, help="HF Hub revision/tag, e.g. step-4000")
    p.add_argument("--no-warmup", action="store_true")
    return Args(**vars(p.parse_args()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    policy = Gr00tServerPolicy(
        args.checkpoint,
        device=args.device,
        default_prompt=args.default_prompt,
        modality_config=args.modality_config,
        revision=args.revision,
    )

    if not args.no_warmup:
        logger.info("Warmup inference...")
        dummy = {
            "state": np.zeros(ACTION_DIM, dtype=np.float32),
            "images": {cam: np.zeros((3, 224, 224), dtype=np.uint8) for cam in EXPECTED_CAMERAS},
            "prompt": args.default_prompt,
        }
        result = policy.infer(dummy)
        logger.info("Warmup done: actions %s in %.0f ms", result["actions"].shape, result["policy_timing"]["infer_ms"])

    websocket_policy_server = _load_websocket_policy_server()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logger.info("Serving GR00T N1.7 on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
