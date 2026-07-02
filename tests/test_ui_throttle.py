from core.ui_throttle import CaptureLogThrottler


def test_capture_log_throttler_batches_messages():
    throttler = CaptureLogThrottler(interval_seconds=1.0)
    throttler.add("screen", "first")
    throttler.add("screen", "second")
    throttler.add("audio", "voice")

    assert not throttler.should_flush(now=0.5)
    assert throttler.should_flush(now=1.1)

    message = throttler.flush(now=1.1)

    assert "audio: 1" in message
    assert "screen: 2" in message
    assert "[audio] voice" in message
    assert not throttler.should_flush(now=2.2)
