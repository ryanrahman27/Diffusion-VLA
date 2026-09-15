import numpy as np

from openpi_client.rtc import action_queue
from openpi_client.rtc import config as rtc_config


def test_action_queue_merge_replaces_with_delay():
    queue = action_queue.ActionQueue(rtc_config.RTCConfig(enabled=True))
    original = np.arange(20, dtype=np.float32).reshape(10, 2)
    processed = original + 100

    queue.merge(original, processed, real_delay=3)

    assert queue.qsize() == 7
    first = queue.get()
    assert np.allclose(first, processed[3])


def test_get_left_over_tracks_consumed_actions():
    queue = action_queue.ActionQueue(rtc_config.RTCConfig(enabled=True))
    original = np.ones((6, 2), dtype=np.float32)
    queue.merge(original, original, real_delay=0)
    queue.get()
    queue.get()
    leftover = queue.get_left_over()
    assert leftover is not None
    assert leftover.shape == (4, 2)
