import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _prepare_modules(monkeypatch):
    dummy_fx = types.ModuleType("facexformer")
    dummy_fx.analyze_face = lambda img: {}
    dummy_fx.extract_embedding = lambda img: b""
    monkeypatch.setitem(sys.modules, "reconhecimento_facial.facexformer", dummy_fx)

    dummy_dem = types.ModuleType("demographics_detection")
    dummy_dem.detect_demographics = lambda img: {}
    monkeypatch.setitem(
        sys.modules, "reconhecimento_facial.demographics_detection", dummy_dem
    )

    dummy_db = types.ModuleType("db")
    dummy_db.get_conn = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda s, exc, val, tb: None
    )
    dummy_db.init_db = lambda: None
    monkeypatch.setitem(sys.modules, "reconhecimento_facial.db", dummy_db)


def import_rec(monkeypatch):
    _prepare_modules(monkeypatch)
    import importlib

    return importlib.import_module("reconhecimento_facial.recognition")


def test_register_person_webcam_social(monkeypatch):
    rec = import_rec(monkeypatch)
    called = {}

    monkeypatch.setattr(rec, "capture_from_webcam", lambda p: "face.jpg")

    def _reg(name, img):
        called["name"] = name
        called["img"] = img
        return True

    monkeypatch.setattr(rec, "register_person", _reg)

    ok = rec.register_person_webcam("Alice")
    assert ok
    assert called["name"] == "Alice"
    assert called["img"] == "face.jpg"
