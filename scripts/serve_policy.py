import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.rtc import config as _rtc_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable Real-Time Chunking (RTC) prefix guidance during flow matching.
    rtc: bool = False
    # Paper Table 4 real-world defaults (pi0.5 bimanual).
    rtc_execution_horizon: int = 25
    rtc_s_min: int = 25
    rtc_max_guidance_weight: float = 5.0
    rtc_num_steps: int = 5

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _rtc_config_from_args(args: Args) -> _rtc_config.RTCConfig | None:
    if not args.rtc:
        return None
    return _rtc_config.RTCConfig(
        enabled=True,
        execution_horizon=args.rtc_execution_horizon,
        s_min=args.rtc_s_min,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=_rtc_config.RTCAttentionSchedule.EXP,
    )


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    rtc_config: _rtc_config.RTCConfig | None = None,
    sample_kwargs: dict | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
            rtc_config=rtc_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    rtc_config = _rtc_config_from_args(args)
    sample_kwargs = {"num_steps": args.rtc_num_steps} if args.rtc else None
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
                rtc_config=rtc_config,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                rtc_config=rtc_config,
                sample_kwargs=sample_kwargs,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    if args.rtc:
        policy_metadata["rtc"] = {
            "enabled": True,
            "execution_horizon": args.rtc_execution_horizon,
            "s_min": args.rtc_s_min,
            "max_guidance_weight": args.rtc_max_guidance_weight,
            "num_steps": args.rtc_num_steps,
            "prefix_mask_end": "H - s (H=model action_horizon, s=rtc_steps_executed or execution_horizon)",
        }

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

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
