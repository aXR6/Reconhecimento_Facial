import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.social_search as ss


def test_run_social_search(monkeypatch, tmp_path):
    db = tmp_path / "db" / "facebook"
    db.mkdir(parents=True)
    (db / "person.jpg").write_text("x")

    dummy_fr = types.ModuleType("face_recognition")
    dummy_fr.load_image_file = lambda p: p
    dummy_fr.face_encodings = lambda img: [np.array([0.0])]
    monkeypatch.setattr(ss, "face_recognition", dummy_fr)

    img = tmp_path / "img.jpg"
    img.write_text("x")

    out = ss.run_social_search([str(img)], sites=["facebook"], db_path=str(db.parent))

    assert (out / "facebook_results.csv").exists()
