import logging
import numpy as np

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
