import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.whisper_translation as wt


def test_translate_file(monkeypatch):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "hello"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    assert wt.translate_file("foo.wav") == "hello"


def test_main_with_expected(monkeypatch, capsys):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "hi"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    wt.main(["--file", "foo.wav", "--expected", "hi"])
    captured = capsys.readouterr()
    assert "hi" in captured.out
    assert "Tradu\u00e7\u00e3o confere" in captured.out
