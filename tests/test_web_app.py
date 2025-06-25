import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.web_app as web


def test_people_route(monkeypatch):
    monkeypatch.setattr(web, "list_people", lambda: ["A"])
    client = web.app.test_client()
    resp = client.get("/people")
    assert resp.json == {"people": ["A"]}


def test_detections_route(monkeypatch):
    monkeypatch.setattr(web, "list_detections", lambda n: [(1, "img", 1, "cap", "mask", "Bob", "2020")])
    client = web.app.test_client()
    resp = client.get("/detections")
    assert resp.json[0]["id"] == 1


def test_recognize_api(monkeypatch, tmp_path):
    dummy = types.ModuleType("rec")
    dummy.recognize_faces = lambda p: ["Bob"]
    monkeypatch.setitem(sys.modules, "reconhecimento_facial.recognition", dummy)
    client = web.app.test_client()
    img = tmp_path / "img.jpg"
    img.write_bytes(b"data")
    with img.open("rb") as fh:
        resp = client.post(
            "/recognize_api",
            data={"image": (fh, "img.jpg")},
            content_type="multipart/form-data",
        )
    assert resp.json == {"names": ["Bob"]}


def test_webcam_page():
    client = web.app.test_client()
    resp = client.get("/webcam")
    assert resp.status_code == 200
