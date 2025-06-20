import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import obstruction_detection as od


def test_detect_obstruction(monkeypatch):
    monkeypatch.setattr(od, "pipeline", lambda *a, **k: lambda img: [{"label": "mask"}])
    od._pipe = None
    assert od.detect_obstruction("any.jpg") == "mask"
