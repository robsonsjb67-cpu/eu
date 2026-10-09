import cv2
import numpy as np

from obs_capture import CaptureStatus, FrameGrabber, RecordedSource, iter_recording


def write_frames(folder, n):
    for i in range(n):
        img = np.full((40, 60, 3), i * 10, np.uint8)
        cv2.imwrite(str(folder / f"f_{i:04d}.png"), img)


def test_iter_recording_reads_all_in_order(tmp_path):
    write_frames(tmp_path, 5)
    vals = [int(img[0, 0, 0]) for img in iter_recording(str(tmp_path))]
    assert vals == [0, 10, 20, 30, 40]
    assert len(list(iter_recording(str(tmp_path), step=2))) == 3


def test_is_recording():
    assert not RecordedSource.is_recording(0)
    assert not RecordedSource.is_recording("1")
    assert not RecordedSource.is_recording("srt://127.0.0.1:9000")
    assert RecordedSource.is_recording("gravacao.mp4")


def test_grabber_plays_recording_then_disconnects(tmp_path):
    write_frames(tmp_path, 4)
    src = RecordedSource(str(tmp_path), fps=200)
    g = FrameGrabber(src, buffer_size=10, disconnect_timeout=0.05, reconnect_delay=0.05).start()
    try:
        last = -1
        for _ in range(4):
            f = g.wait_for_frame(last, timeout=2)
            assert f is not None
            last = f.index
        assert last == 3
        f = g.wait_for_frame(last, timeout=0.5)
        assert f is None
        assert g.status == CaptureStatus.DISCONNECTED
    finally:
        g.stop()


def test_save_screenshot(tmp_path):
    from obs_capture import save_screenshot

    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    path = save_screenshot(img, str(tmp_path / "prints"))
    assert path.endswith(".png") and np.array_equal(cv2.imread(path), img)
