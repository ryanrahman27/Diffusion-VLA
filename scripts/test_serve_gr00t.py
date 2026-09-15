"""Tests for serve_gr00t.py.

1. Protocol test (default, no GPU/gr00t install needed) — real openpi-client stack
   against a stub policy with GR00T's 16-step horizon:
     uv run scripts/test_serve_gr00t.py

2. Checkpoint test (GPU box, Isaac-GR00T uv env):
     python scripts/test_serve_gr00t.py --checkpoint /path/to/checkpoint-1000
"""

import argparse
import threading
import time

import numpy as np

from serve_gr00t import ACTION_DIM, ACTION_HORIZON, EXPECTED_CAMERAS, _load_websocket_policy_server


def make_piperx_obs() -> dict:
    return {
        "state": np.random.uniform(-1, 1, size=(ACTION_DIM,)).astype(np.float32),
        "images": {cam: np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8) for cam in EXPECTED_CAMERAS},
        "prompt": "pick towel from pile, fold and stack",
    }


class StubPolicy:
    metadata = {"policy": "gr00t-stub", "n_action_steps": ACTION_HORIZON, "action_dim": ACTION_DIM}

    def infer(self, obs: dict, *, noise=None) -> dict:  # noqa: ARG002
        assert set(obs["images"]) == set(EXPECTED_CAMERAS), sorted(obs["images"])
        assert np.asarray(obs["state"]).shape == (ACTION_DIM,)
        actions = np.tile(np.arange(ACTION_HORIZON, dtype=np.float32)[:, None], (1, ACTION_DIM))
        return {"actions": actions, "policy_timing": {"infer_ms": 0.1}}

    def reset(self) -> None:
        pass


def run_protocol_test(port: int = 18766) -> None:
    from openpi_client import action_chunk_broker, websocket_client_policy

    websocket_policy_server = _load_websocket_policy_server()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=StubPolicy(), host="127.0.0.1", port=port, metadata=StubPolicy.metadata
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.5)

    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
    horizon = client.get_server_metadata()["n_action_steps"]
    assert horizon == ACTION_HORIZON, horizon

    result = client.infer(make_piperx_obs())
    assert result["actions"].shape == (ACTION_HORIZON, ACTION_DIM), result["actions"].shape

    broker = action_chunk_broker.ActionChunkBroker(client, action_horizon=horizon)
    for step in range(ACTION_HORIZON + 3):
        action = broker.infer(make_piperx_obs())["actions"]
        assert action.shape == (ACTION_DIM,), action.shape
        assert float(action[0]) == step % ACTION_HORIZON, (step, action[0])

    print("PROTOCOL TEST PASSED: GR00T 16-step chunks flow through the PiperX client stack.")


def run_checkpoint_test(checkpoint: str, device: str) -> None:
    from serve_gr00t import Gr00tServerPolicy

    policy = Gr00tServerPolicy(checkpoint, device=device, default_prompt="pick towel from pile, fold and stack")
    t0 = time.monotonic()
    result = policy.infer(make_piperx_obs())
    dt = time.monotonic() - t0
    actions = result["actions"]
    assert actions.shape == (ACTION_HORIZON, ACTION_DIM), actions.shape
    assert np.isfinite(actions).all()
    print(f"CHECKPOINT TEST PASSED: actions {actions.shape} in {dt:.2f}s")
    print("  first action:", np.round(actions[0], 4))
    print("  grippers L/R:", np.round(actions[:3, 6], 4), np.round(actions[:3, 13], 4), "(expect ~0..0.07 m)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.checkpoint:
        run_checkpoint_test(args.checkpoint, args.device)
    else:
        run_protocol_test()
