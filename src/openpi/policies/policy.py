from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.rtc import config as _rtc_config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

_RTC_PREV_CHUNK_KEY = "rtc_prev_chunk_left_over"
_RTC_INFERENCE_DELAY_KEY = "rtc_inference_delay"
_RTC_STEPS_EXECUTED_KEY = "rtc_steps_executed"

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_config: _rtc_config.RTCConfig | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_config = rtc_config

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if hasattr(model, "sample_actions_with_rtc"):
                self._sample_actions_with_rtc = nnx_utils.module_jit(
                    model.sample_actions_with_rtc,
                    static_argnames=("inference_delay", "rtc_config", "num_steps", "steps_executed"),
                )
            else:
                self._sample_actions_with_rtc = None
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        inference_delay: int | None = None,
        steps_executed: int | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_prev_chunk = prev_chunk_left_over
        rtc_inference_delay = inference_delay
        rtc_steps_executed = steps_executed
        if rtc_prev_chunk is None and _RTC_PREV_CHUNK_KEY in inputs:
            rtc_prev_chunk = inputs.pop(_RTC_PREV_CHUNK_KEY)
        if rtc_inference_delay is None and _RTC_INFERENCE_DELAY_KEY in inputs:
            rtc_inference_delay = int(inputs.pop(_RTC_INFERENCE_DELAY_KEY))
        if rtc_steps_executed is None and _RTC_STEPS_EXECUTED_KEY in inputs:
            rtc_steps_executed = int(inputs.pop(_RTC_STEPS_EXECUTED_KEY))
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        use_rtc = (
            self._rtc_config is not None
            and self._rtc_config.enabled
            and rtc_prev_chunk is not None
        )
        if use_rtc and self._is_pytorch_model:
            sample_kwargs["rtc_config"] = self._rtc_config
            sample_kwargs["prev_chunk_left_over"] = rtc_prev_chunk
            sample_kwargs["inference_delay"] = int(rtc_inference_delay or 0)
            if rtc_steps_executed is not None:
                sample_kwargs["steps_executed"] = int(rtc_steps_executed)
        elif use_rtc and not self._is_pytorch_model:
            if self._sample_actions_with_rtc is None:
                logging.warning("RTC requested but model has no sample_actions_with_rtc; falling back.")
                use_rtc = False
            else:
                sample_kwargs["rtc_config"] = self._rtc_config
                sample_kwargs["prev_chunk_left_over"] = jnp.asarray(rtc_prev_chunk)
                sample_kwargs["inference_delay"] = int(rtc_inference_delay or 0)
                if rtc_steps_executed is not None:
                    sample_kwargs["steps_executed"] = int(rtc_steps_executed)
        elif rtc_prev_chunk is not None:
            logging.warning("RTC prev_chunk_left_over provided but rtc_config is disabled; ignoring.")

        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if use_rtc and not self._is_pytorch_model:
            model_actions = self._sample_actions_with_rtc(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
        else:
            model_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        outputs = {
            "state": inputs["state"],
            "actions": model_actions,
        }
        if "state_absolute" in inputs:
            # Leave batched; shared unbatch below applies [0, ...] once.
            outputs["state_absolute"] = inputs["state_absolute"]
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        rtc_enabled = self._rtc_config is not None and self._rtc_config.enabled
        actions_model = np.asarray(outputs["actions"]).copy() if rtc_enabled else None

        outputs = self._output_transform(outputs)
        if actions_model is not None:
            # Keep normalized model-space chunk for RTC client bookkeeping. Output transforms
            # (e.g. PiperXOutputs) only return robot-facing keys and would drop this field.
            outputs["actions_model"] = actions_model
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
