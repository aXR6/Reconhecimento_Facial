import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.whisper_translation as wt
wt.hf_pipeline = object()


def test_translate_file(monkeypatch):
    dummy_pipe = lambda *a, **k: {"text": "hi"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    assert wt.translate_file("foo.wav", source_lang="pt", target_lang="en") == "hi"


def test_translate_file_other_lang(monkeypatch):
    dummy_pipe = lambda *a, **k: {"text": "hola"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    assert wt.translate_file("foo.wav", source_lang="es", target_lang="fr") == ""


def test_transcribe_file(monkeypatch):
    dummy_pipe = lambda *a, **k: {"text": "oi"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    assert wt.transcribe_file("foo.wav", source_lang="pt") == "oi"


def test_whisper_translate_file(monkeypatch):
    dummy_pipe = lambda *a, **k: {"text": "hi"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    assert (
        wt.whisper_translate_file("foo.wav", model_name="base", source_lang="pt", target_lang="en")
        == "hi"
    )


def test_main_with_expected(monkeypatch, capsys):
    dummy_pipe = lambda *a, **k: {"text": "hi"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    wt.main(["--file", "foo.wav", "--expected", "hi", "--src", "pt", "--dst", "en"])
    captured = capsys.readouterr()
    assert "hi" in captured.out
    assert "Tradu\u00e7\u00e3o confere" in captured.out


def test_main_transcribe_flag(monkeypatch, capsys):
    dummy_pipe = lambda *a, **k: {"text": "hello"}
    monkeypatch.setattr(wt, "_get_pipe", lambda *a, **k: dummy_pipe)
    wt.main(["--file", "foo.wav", "--transcribe", "--src", "pt", "--dst", "en"])
    captured = capsys.readouterr()
    assert "hello" in captured.out

