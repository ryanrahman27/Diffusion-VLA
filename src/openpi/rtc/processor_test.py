"""Unit tests for RTC prefix weight schedules."""

import pytest
import torch

from openpi.rtc import config as rtc_config
from openpi.rtc.processor import RTCProcessor


@pytest.mark.parametrize(
    ("schedule", "start", "end", "total"),
    [
        (rtc_config.RTCAttentionSchedule.ZEROS, 3, 8, 10),
        (rtc_config.RTCAttentionSchedule.ONES, 3, 8, 10),
        (rtc_config.RTCAttentionSchedule.LINEAR, 3, 8, 10),
        (rtc_config.RTCAttentionSchedule.EXP, 3, 8, 10),
    ],
)
def test_get_prefix_weights_shape(schedule, start, end, total):
    processor = RTCProcessor(
        rtc_config.RTCConfig(prefix_attention_schedule=schedule, execution_horizon=end)
    )
    weights = processor.get_prefix_weights(start, end, total)
    assert weights.shape == (total,)


def test_zeros_schedule_masks_prefix_region():
    processor = RTCProcessor(
        rtc_config.RTCConfig(prefix_attention_schedule=rtc_config.RTCAttentionSchedule.ZEROS)
    )
    weights = processor.get_prefix_weights(4, 8, 10)
    assert torch.all(weights[:4] == 1.0)
    assert torch.all(weights[4:] == 0.0)


def test_denoise_step_without_prev_chunk_is_passthrough():
    processor = RTCProcessor(rtc_config.RTCConfig())
    x_t = torch.zeros(1, 5, 3)

    def original(x):
        return torch.ones_like(x)

    v_t = processor.denoise_step(x_t, None, 2, 0.5, original)
    assert torch.allclose(v_t, torch.ones_like(x_t))
