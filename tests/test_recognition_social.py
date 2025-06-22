import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _prepare_modules(monkeypatch):
    dummy_fx = types.ModuleType("facexformer")
    dummy_fx.analyze_face = lambda img: {}
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


def test_recognize_faces_social(monkeypatch):
    rec = import_rec(monkeypatch)
    called = {}
    monkeypatch.setattr(rec, "recognize_faces", lambda img: ["Alice"])

    def dummy_thread(target, args=(), kwargs=None, daemon=None):
        called["args"] = args
        target(*args, **(kwargs or {}))
        return types.SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(rec.threading, "Thread", dummy_thread)
    monkeypatch.setattr(
        rec, "_social_search_background", lambda *a, **k: called.update({"bg": a})
    )

    names = rec.recognize_faces_social("img.jpg", sites=["facebook"])
    assert names == ["Alice"]
    assert called["bg"][1] == "Alice"
