import numpy as np

from obs_capture import CaptureStatus, FrameGrabber


class FakeSource:
    def open(self): return True
    def read(self): return False, None
    def close(self): pass


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def make(clock, **kw):
    return FrameGrabber(FakeSource(), buffer_size=3, clock=clock, **kw)


def noise(seed):
    return np.random.default_rng(seed).integers(0, 255, (72, 128, 3), dtype=np.uint8)


def test_buffer_keeps_only_recent_frames():
    clock = Clock()
    g = make(clock)
    for i in range(10):
        clock.t = i * 0.1
        g.push(noise(i))
    assert [f.index for f in g.recent()] == [7, 8, 9]
    assert g.latest().index == 9
    assert g.stats.frames_dropped == 7


def test_frozen_frames_detected_and_recovered():
    clock = Clock()
    g = make(clock, freeze_seconds=1.0)
    img = noise(1)
    for i in range(15):
        clock.t = i * 0.1
        g.push(img)
    assert g.status == CaptureStatus.FROZEN
    clock.t = 1.6
    g.push(noise(2))
    assert g.status == CaptureStatus.CONNECTED


def test_disconnect_when_frames_stop():
    clock = Clock()
    g = make(clock, disconnect_timeout=2.0)
    g.push(noise(1))
    assert g.status == CaptureStatus.CONNECTED
    clock.t = 3.0
    g.latest()
    assert g.status == CaptureStatus.DISCONNECTED
