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
