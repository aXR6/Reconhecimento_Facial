import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.social_search as ss


def test_run_social_search(monkeypatch, tmp_path):
    called = {}

    def dummy_clone(path):
        return tmp_path

    def dummy_run(cmd, check, cwd=None):
        called['cmd'] = cmd
        called['cwd'] = cwd

    monkeypatch.setattr(ss, 'ensure_repo', dummy_clone)
    monkeypatch.setattr(ss.subprocess, 'run', dummy_run)

    img = tmp_path / "img.jpg"
    img.write_text("dummy")

    out = ss.run_social_search([str(img)], sites=['facebook'])

    assert 'social_mapper.py' in called['cmd'][1]
    assert called['cwd'] == str(tmp_path)
    assert out == tmp_path / 'SM-Results'

