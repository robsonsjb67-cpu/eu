"""Captura por prints do obs-websocket v5, com um OBS simulado."""
import base64
import json

import cv2
import numpy as np

from obs_capture import CaptureStatus, FrameGrabber, OBSScreenshotSource, save_screenshot, source_from_config
from config import DEFAULT_CONFIG, deep_merge


class FakeOBS:
    """Responde como o OBS: Hello → Identify → Identified → requests."""

    def __init__(self, password=None, scene="Cena Tibia", fail_source=False):
        self.password, self.scene, self.fail_source = password, scene, fail_source
        self.out, self.requests, self.closed = [], [], False
        hello = {"rpcVersion": 1}
        if password:
            hello["authentication"] = {"salt": "sal", "challenge": "desafio"}
        self.out.append({"op": 0, "d": hello})
        self.counter = 0

    def __call__(self, url, timeout=None):
        assert url == "ws://localhost:4455"
        return self

    def send(self, text):
        msg = json.loads(text)
        op, d = msg["op"], msg["d"]
        if op == 1:
            ok = not self.password or d.get("authentication") == OBSScreenshotSource.auth_string(
                self.password, "sal", "desafio")
            if ok:
                self.out.append({"op": 2, "d": {"negotiatedRpcVersion": 1}})
            else:
                self.out.append({"op": -1, "d": {}})  # o OBS real fecha a conexão
            return
        assert op == 6
        self.requests.append(d)
        # um evento qualquer no meio, que deve ser ignorado
        self.out.append({"op": 5, "d": {"eventType": "SceneItemSelected"}})
        rt, rid = d["requestType"], d["requestId"]
        if rt == "GetCurrentProgramScene":
            resp = {"currentProgramSceneName": self.scene}
            ok = True
        elif rt == "GetSourceScreenshot":
            ok = not self.fail_source
            self.counter += 1
            img = np.full((90, 160, 3), self.counter * 20 % 255, np.uint8)
            _, png = cv2.imencode(".png", img)
            resp = {"imageData": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}
        self.out.append({"op": 7, "d": {"requestType": rt, "requestId": rid,
                                        "requestStatus": {"result": ok, "code": 100 if ok else 600,
                                                          "comment": None if ok else "fonte não existe"},
                                        "responseData": resp if ok else None}})

    def recv(self):
        return json.dumps(self.out.pop(0))

    def close(self):
        self.closed = True


def src(fake, url="obsws://localhost:4455"):
    return OBSScreenshotSource.from_url(url, connect=fake, sleep=lambda s: None, fps=1000)


def test_url_parsing():
    s = OBSScreenshotSource.from_url("obsws://segredo@192.168.0.5:4460/Captura%20Tibia")
    assert (s.host, s.port, s.password, s.source_name) == ("192.168.0.5", 4460, "segredo", "Captura Tibia")
    s = OBSScreenshotSource.from_url("obsws://localhost", password="cfg", source_name=None)
    assert (s.port, s.password, s.source_name) == (4455, "cfg", None)
    cfg = deep_merge(DEFAULT_CONFIG, {"capture": {"source": "obsws://localhost/X", "obs_password": "p"}})
    s = source_from_config(cfg)
    assert isinstance(s, OBSScreenshotSource) and s.password == "p" and s.source_name == "X"


def test_auth_string_matches_protocol_example():
    # Exemplo da documentação do obs-websocket v5.
    got = OBSScreenshotSource.auth_string("supersecretpassword",
                                          "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI=",
                                          "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY=")
    assert got == "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4="


def test_screenshot_with_password_and_current_scene():
    fake = FakeOBS(password="segredo")
    s = OBSScreenshotSource.from_url("obsws://segredo@localhost:4455", connect=fake,
                                     sleep=lambda x: None, fps=1000)
    assert s.open() and s.source_name == "Cena Tibia"
    ok, img = s.read()
    assert ok and img.shape == (90, 160, 3)
    req = fake.requests[-1]
    assert req["requestType"] == "GetSourceScreenshot"
    assert req["requestData"]["sourceName"] == "Cena Tibia"


def test_wrong_or_missing_password_fails_cleanly():
    fake = FakeOBS(password="segredo")
    s = OBSScreenshotSource.from_url("obsws://errada@localhost:4455", connect=fake, sleep=lambda x: None)
    assert not s.open() and "senha" in s.last_error
    s = src(FakeOBS(password="segredo"))
    assert not s.open() and "exige senha" in s.last_error


def test_missing_source_reports_error():
    s = src(FakeOBS(fail_source=True), "obsws://localhost:4455/NaoExiste")
    assert s.open()
    ok, img = s.read()
    assert not ok and "fonte não existe" in s.last_error


def test_grabber_receives_screenshots():
    g = FrameGrabber(src(FakeOBS()), buffer_size=3).start()
    try:
        f = g.wait_for_frame(-1, timeout=2)
        f2 = g.wait_for_frame(f.index, timeout=2)
        assert f2 is not None and g.status == CaptureStatus.CONNECTED
    finally:
        g.stop()


def test_save_screenshot(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    path = save_screenshot(img, str(tmp_path / "prints"))
    assert path.endswith(".png")
    assert np.array_equal(cv2.imread(path), img)          # PNG sem perdas
