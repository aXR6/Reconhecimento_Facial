import os
import sys
import numpy as np
import cv2
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

dummy_fx = types.ModuleType("facexformer")
dummy_fx.analyze_face = lambda img: {}
sys.modules["reconhecimento_facial.facexformer"] = dummy_fx
dummy_dem = types.ModuleType("demographics_detection")
dummy_dem.detect_demographics = lambda img: {}
sys.modules["reconhecimento_facial.demographics_detection"] = dummy_dem

from reconhecimento_facial.face_detection import detect_faces
import reconhecimento_facial.face_detection as fd_mod


def test_detect_no_faces(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(img_path), img)
    count = detect_faces(str(img_path), str(tmp_path / "out.jpg"), use_hf=False)
    assert count == 0


def test_detect_hf_increment_mediapipe(monkeypatch, tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(img_path), img)

    class DummyFD:
        def __init__(self, *a, **k):
            pass

        def process(self, _):
            bbox = type("bbox", (), {"xmin": 0, "ymin": 0, "width": 0.1, "height": 0.1})
            loc = type("loc", (), {"relative_bounding_box": bbox()})
            det = type("det", (), {"location_data": loc()})
            return type("res", (), {"detections": [det()]})()

    dummy_mp = type(
        "mp",
        (),
        {
            "solutions": type(
                "sol",
                (),
                {"face_detection": type("fd", (), {"FaceDetection": DummyFD})},
            )(),
        },
    )
    monkeypatch.setattr(fd_mod, "mp", dummy_mp)

    count = fd_mod.detect_faces(
        str(img_path), str(tmp_path / "out.jpg"), use_hf=True, hf_model="mediapipe"
    )
    assert count == 1


def test_detect_hf_increment_yolov8(monkeypatch, tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(img_path), img)

    class DummyXY:
        def __init__(self):
            self.arr = np.array([[0, 0, 10, 10]])

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    class DummyBoxes:
        def __init__(self):
            self.xyxy = DummyXY()

        def __len__(self):
            return 1

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()

    class DummyYOLO:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return [DummyResult()]

    monkeypatch.setattr(fd_mod, "YOLO", DummyYOLO)
    monkeypatch.setattr(fd_mod, "hf_hub_download", lambda *a, **k: "")

    count = fd_mod.detect_faces(
        str(img_path), str(tmp_path / "out.jpg"), use_hf=True, hf_model="yolov8"
    )
    assert count == 1


