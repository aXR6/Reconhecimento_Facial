import logging
import numpy as np
import sys
import types
from importlib.util import find_spec


def _patch_face_recognition_models() -> None:
    """Replace the face_recognition_models package to avoid deprecated
    pkg_resources import used by the original package."""

    if "face_recognition_models" in sys.modules:
        return

    spec = find_spec("face_recognition_models")
    if not spec or not spec.submodule_search_locations:
        return

    base_path = (spec.submodule_search_locations[0])
    def _resource(filename: str) -> str:
        return f"{base_path}/models/{filename}"

    module = types.ModuleType("face_recognition_models")
    module.pose_predictor_model_location = (
        lambda: _resource("shape_predictor_68_face_landmarks.dat")
    )
    module.pose_predictor_five_point_model_location = (
        lambda: _resource("shape_predictor_5_face_landmarks.dat")
    )
    module.face_recognition_model_location = (
        lambda: _resource("dlib_face_recognition_resnet_model_v1.dat")
    )
    module.cnn_face_detector_model_location = (
        lambda: _resource("mmod_human_face_detector.dat")
    )

    sys.modules["face_recognition_models"] = module


_patch_face_recognition_models()

try:
    import face_recognition
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    face_recognition = None

from db import get_conn

logger = logging.getLogger(__name__)


def register_person(name: str, image_path: str) -> None:
    if face_recognition is None:
        logger.error("face_recognition not installed")
        return
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        logger.error("No face found in %s", image_path)
        return
    encoding = encodings[0]
    with get_conn() as conn:
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO people (name, embedding) VALUES (%s, %s)",
            (name, encoding.tobytes()),
        )
        conn.commit()


def recognize_faces(image_path: str) -> list[str]:
    if face_recognition is None:
        logger.error("face_recognition not installed")
        return []
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        return []
    with get_conn() as conn:
        if conn is None:
            return []
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM people")
        data = cur.fetchall()
    known_names = [row[0] for row in data]
    known_encodings = [np.frombuffer(row[1], dtype=np.float64) for row in data]
    recognized = []
    for face_enc in encodings:
        if not known_encodings:
            continue
        dists = face_recognition.face_distance(known_encodings, face_enc)
        best = dists.argmin()
        if dists[best] < 0.6:
            recognized.append(known_names[best])
    return recognized
