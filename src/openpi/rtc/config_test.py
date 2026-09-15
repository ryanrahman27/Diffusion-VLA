from openpi.rtc import config as rtc_config


def test_prefix_attention_end_matches_kinetix():
    cfg = rtc_config.RTCConfig(execution_horizon=25)
    # H=50, s=25 -> mask end = 25 (kinetix: prefix_attention_horizon = H - execute_horizon)
    assert cfg.prefix_attention_end(steps_executed=25, action_horizon=50) == 25
    assert cfg.prefix_attention_end(steps_executed=None, action_horizon=50) == 25


def test_prefix_attention_end_uses_dynamic_s():
    cfg = rtc_config.RTCConfig(execution_horizon=10)
    assert cfg.prefix_attention_end(steps_executed=4, action_horizon=50) == 46
