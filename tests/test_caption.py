import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import llm_service


def test_generate_caption(monkeypatch):
    monkeypatch.setattr(llm_service, "pipeline", lambda *a, **k: lambda img: [{"generated_text": "ok"}])
    llm_service._pipe = None
    assert llm_service.generate_caption("any.jpg") == "ok"
