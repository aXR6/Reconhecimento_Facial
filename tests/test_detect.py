import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from face_detection import detect_faces


def test_detect_no_faces(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(img_path), img)
    count = detect_faces(str(img_path), str(tmp_path / "out.jpg"), use_hf=False)
    assert count == 0
