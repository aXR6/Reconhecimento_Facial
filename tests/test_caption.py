import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.llm_service as llm_service
import types


def test_generate_caption(monkeypatch):
    monkeypatch.setattr(llm_service, "pipeline", lambda *a, **k: lambda img: [{"generated_text": "ok"}])
    dummy = types.SimpleNamespace(detect_demographics=lambda img: {})
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.demographics_detection', dummy)
    llm_service._pipe = None
    assert llm_service.generate_caption("any.jpg") == "ok"


def test_generate_caption_with_demo(monkeypatch):
    monkeypatch.setattr(llm_service, "pipeline", lambda *a, **k: lambda img: [{"generated_text": "a person"}])
    info = {
        "gender": "female",
        "age": "20-29",
        "ethnicity": "asian",
        "skin": "light",
    }
    dummy = types.SimpleNamespace(detect_demographics=lambda img: info)
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.demographics_detection', dummy)
    llm_service._pipe = None
    caption = llm_service.generate_caption("any.jpg")
    assert "a person" in caption
    assert "female" in caption
