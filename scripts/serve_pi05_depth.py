"""Serve a trained π₀.₅ + depth policy over WebSocket (PiperX depth extension)."""

import dataclasses
import logging
import socket

import tyro

from openpi.depth import train_config as depth_config
from openpi.depth.policy_config import create_trained_depth_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Checkpoint:
    """Load a π₀.₅+depth policy from a JAX checkpoint directory."""

    # Training config name from ``openpi.depth.train_config`` (e.g. pi05_piperx_flatten_depth).
    config: str
    # Checkpoint step dir (e.g. checkpoints/pi05_piperx_flatten_depth/exp/10000).
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for serve_pi05_depth.py."""

    # Policy checkpoint to serve.
    policy: Checkpoint = dataclasses.field(
        default_factory=lambda: Checkpoint(
            config="pi05_piperx_flatten_depth",
            dir="checkpoints/pi05_piperx_flatten_depth/flatten_depth_v1/10000",
        )
    )

    # Used when the observation has no ``prompt`` key.
    default_prompt: str | None = "flatten towel"

    # WebSocket port.
    port: int = 8000

    # Save observations/actions for debugging.
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    # Validate config name early for a clear error message.
    depth_config.get_config(args.policy.config)
    return create_trained_depth_policy(
        args.policy.config,
        args.policy.dir,
        default_prompt=args.default_prompt,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating π₀.₅+depth server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Expect obs keys: images, depth, state; optional prompt")
    logging.info("  images/depth camera keys: cam_front, cam_left_wrist, cam_right_wrist")
    logging.info("  depth: uint16 Z16 mm (H,W) or (1,H,W); see openpi.depth.piperx_depth_policy")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
