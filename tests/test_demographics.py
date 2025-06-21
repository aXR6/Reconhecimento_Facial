import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.demographics_detection as dd


def test_detect_demographics_facexformer(monkeypatch):
    dummy = types.ModuleType('dummy')
    dummy.detect_demographics = lambda img: {
        'gender': 'male', 'age': '30', 'ethnicity': 'asian', 'skin': 'light'
    }
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.facexformer.inference', dummy)
    dd.set_backend('facexformer')
    result = dd.detect_demographics('any.jpg')
    assert result == {
        'gender': 'male',
        'age': '30',
        'ethnicity': 'asian',
        'skin': 'light'
    }


def test_detect_demographics_deepface(monkeypatch):
    dummy = types.ModuleType('dummy2')
    dummy.detect_demographics = lambda img: {
        'gender': 'female', 'age': '25', 'ethnicity': 'white', 'skin': 'fair'
    }
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.deepface_integration', dummy)
    dd.set_backend('deepface')
    result = dd.detect_demographics('any.jpg')
    assert result == {
        'gender': 'female',
        'age': '25',
        'ethnicity': 'white',
        'skin': 'fair'
    }
