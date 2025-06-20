import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.demographics_detection as dd


def test_detect_demographics(monkeypatch):
    monkeypatch.setattr(
        dd,
        "pipeline",
        lambda *a, **k: lambda img: [
            {"label": "gender: male"},
            {"label": "age: 30"},
            {"label": "ethnicity: asian"},
            {"label": "skin: light"},
        ],
    )
    dd._pipe = None
    result = dd.detect_demographics("any.jpg")
    assert result == {
        "gender": "male",
        "age": "30",
        "ethnicity": "asian",
        "skin": "light",
    }
