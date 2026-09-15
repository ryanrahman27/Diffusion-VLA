"""Real-Time Chunking (RTC) client configuration."""

from __future__ import annotations

import dataclasses
import enum


class RTCAttentionSchedule(enum.Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclasses.dataclass
class RTCConfig:
    """Client RTC configuration.

    ``execution_horizon`` is paper **s** (steps executed per chunk before re-infer).
    Prefix mask end on the server is ``H - s`` (Eq. 5).
    """

    enabled: bool = True
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    max_guidance_weight: float = 5.0
    execution_horizon: int = 25
    s_min: int = 25

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
        if self.s_min <= 0:
            raise ValueError(f"s_min must be positive, got {self.s_min}")
