import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.whisper_translation as wt


def test_translate_file(monkeypatch):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "ola"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    monkeypatch.setattr(wt, "_translate_text", lambda t, s, d: "hello")
    assert wt.translate_file("foo.wav", source_lang="pt", target_lang="en") == "hello"


def test_whisper_translate_file(monkeypatch):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "hi"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    assert wt.whisper_translate_file("foo.wav", model_name="base", source_lang="pt") == "hi"


def test_main_with_expected(monkeypatch, capsys):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "oi"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    monkeypatch.setattr(wt, "_translate_text", lambda t, s, d: "hi")
    wt.main(["--file", "foo.wav", "--expected", "hi", "--src", "pt", "--tgt", "en"])
    captured = capsys.readouterr()
    assert "hi" in captured.out
    assert "Tradu\u00e7\u00e3o confere" in captured.out


def test_main_whisper_flag(monkeypatch, capsys):
    dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"text": "hello"})
    monkeypatch.setattr(wt, "whisper", types.SimpleNamespace(load_model=lambda n: dummy_model))
    wt.main(["--file", "foo.wav", "--whisper", "--src", "pt"])
    captured = capsys.readouterr()
    assert "hello" in captured.out

