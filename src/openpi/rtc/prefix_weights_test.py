import jax.numpy as jnp

from openpi.rtc import config as rtc_config
from openpi.rtc.prefix_weights import get_prefix_weights_jax
from openpi.rtc.prefix_weights import get_prefix_weights_np


def test_prefix_weights_jax_matches_numpy():
    weights_np = get_prefix_weights_np(4, 8, 10, rtc_config.RTCAttentionSchedule.EXP)
    weights_jax = get_prefix_weights_jax(4, 8, 10, rtc_config.RTCAttentionSchedule.EXP)
    assert jnp.allclose(weights_jax, jnp.asarray(weights_np))
