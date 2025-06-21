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
    result = dd.detect_demographics('any.jpg')
    assert result == {
        'gender': 'male',
        'age': '30',
        'ethnicity': 'asian',
        'skin': 'light'
    }


def test_detect_demographics_array(monkeypatch):
    arr = object()
    dummy = types.ModuleType('dummy')

    def _detect(img):
        assert img is arr
        return {'gender': 'female'}

    dummy.detect_demographics = _detect
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.facexformer.inference', dummy)
    result = dd.detect_demographics(arr)
    assert result == {'gender': 'female'}


def test_analyze_face(monkeypatch):
    dummy = types.ModuleType('dummy')

    def _analyze(img):
        return {
            'age': '20-29',
            'gender': 'female',
            'ethnicity': 'white',
            'landmarks': [[0, 0]],
            'headpose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
            'segmentation': [[0]],
            'attributes': {'Smiling': True},
            'visibility': 1,
        }

    dummy.analyze_face = _analyze
    dummy.detect_demographics = lambda img: {}
    monkeypatch.setitem(sys.modules, 'reconhecimento_facial.facexformer.inference', dummy)
    from reconhecimento_facial.facexformer import analyze_face

    result = analyze_face('img.jpg')
    assert result['gender'] == 'female'
    assert 'landmarks' in result
