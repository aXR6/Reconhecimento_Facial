import logging
import numpy as np
import sys
import types
from importlib.util import find_spec

import cv2
import os

if __package__ is None or __package__ == "":
    import pathlib
    import sys as _sys

    _sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"


def _patch_face_recognition_models() -> None:
    """Replace the face_recognition_models package to avoid deprecated
    pkg_resources import used by the original package."""

    if "face_recognition_models" in sys.modules:
        return

    spec = find_spec("face_recognition_models")
    if not spec or not spec.submodule_search_locations:
        return

    base_path = spec.submodule_search_locations[0]

    def _resource(filename: str) -> str:
        return f"{base_path}/models/{filename}"

    module = types.ModuleType("face_recognition_models")
    module.pose_predictor_model_location = lambda: _resource(
        "shape_predictor_68_face_landmarks.dat"
    )
    module.pose_predictor_five_point_model_location = lambda: _resource(
        "shape_predictor_5_face_landmarks.dat"
    )
    module.face_recognition_model_location = lambda: _resource(
        "dlib_face_recognition_resnet_model_v1.dat"
    )
    module.cnn_face_detector_model_location = lambda: _resource("mmod_human_face_detector.dat")

    sys.modules["face_recognition_models"] = module


_patch_face_recognition_models()

try:
    import face_recognition
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    face_recognition = None

from reconhecimento_facial.db import get_conn, init_db
from reconhecimento_facial.demographics_detection import detect_demographics

logger = logging.getLogger(__name__)


def capture_from_webcam(tmp_path: str) -> bool:
    """Open webcam preview and capture a frame on key press."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Nao foi possivel acessar a webcam")
        return False

    captured = False
    print("Pressione 'c' ou ESPAÇO para capturar, 'q' para cancelar")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Falha ao capturar imagem da webcam")
            break
        cv2.imshow("webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("c"), ord(" ")):
            cv2.imwrite(tmp_path, frame)
            captured = True
            break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return captured


def register_person(name: str, image_path: str) -> bool:
    """Register a person in the database.

    Returns ``True`` if the operation succeeds. ``False`` is returned when a
    face cannot be encoded or the database is unavailable. This allows callers
    to properly report failures instead of always signalling success.
    """
    if face_recognition is None:
        logger.error("face_recognition not installed")
        return False
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        logger.error("No face found in %s", image_path)
        return False
    encoding = encodings[0]
    init_db()
    with get_conn() as conn:
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO people (name, embedding) VALUES (%s, %s)",
            (name, encoding.tobytes()),
        )
        conn.commit()
    return True


def register_person_webcam(name: str) -> bool:
    """Capture an image from the webcam and register the person.

    The function returns ``True`` only when both the capture and the database
    insertion succeed. This behaviour ensures that callers do not display a
    successful message when the registration actually failed.
    """
    tmp = f"/tmp/{name.replace(' ', '_')}.jpg"
    if not capture_from_webcam(tmp):
        return False
    try:
        ok = register_person(name, tmp)
        if ok:
            print("Cadastro salvo com sucesso")
        return ok
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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


def recognize_webcam() -> None:
    """Captura a webcam e exibe rostos identificados em tempo real."""
    if face_recognition is None:
        logger.error("face_recognition not installed")
        return

    with get_conn() as conn:
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM people")
        data = cur.fetchall()

    known_names = [row[0] for row in data]
    known_encodings = [np.frombuffer(row[1], dtype=np.float64) for row in data]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR captured by OpenCV to RGB for face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        for (top, right, bottom, left), face_enc in zip(locations, encodings):
            name = "Unknown"
            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, face_enc)
                best = dists.argmin()
                if dists[best] < 0.6:
                    name = known_names[best]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            crop = frame[top:bottom, left:right]
            info = ""
            try:
                dem = detect_demographics(crop)
                gender = dem.get("gender")
                age = dem.get("age")
                ethnicity = dem.get("ethnicity")
                skin = dem.get("skin")
                parts = [name]
                if gender:
                    parts.append(gender)
                if age:
                    parts.append(age)
                if ethnicity:
                    parts.append(ethnicity)
                if skin:
                    parts.append(skin)
                info = ", ".join(parts)
            except Exception as exc:  # noqa: BLE001
                logger.error("demographics error: %s", exc)
                info = name
            cv2.putText(
                frame,
                info,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def recognize_webcam_mediapipe() -> None:
    """Captura a webcam usando MediaPipe para detecção e identifica rostos."""
    if face_recognition is None:
        logger.error("face_recognition not installed")
        return
    try:
        import mediapipe as mp  # type: ignore
    except ModuleNotFoundError:
        logger.error("mediapipe not installed")
        return

    mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0)

    with get_conn() as conn:
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM people")
        data = cur.fetchall()

    known_names = [row[0] for row in data]
    known_encodings = [np.frombuffer(row[1], dtype=np.float64) for row in data]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_fd.process(rgb)
        locations = []
        if res.detections:
            for det in res.detections:
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * frame.shape[1])
                y = int(box.ymin * frame.shape[0])
                w = int(box.width * frame.shape[1])
                h = int(box.height * frame.shape[0])
                locations.append((y, x + w, y + h, x))
        encodings = face_recognition.face_encodings(rgb, locations)
        for (top, right, bottom, left), face_enc in zip(locations, encodings):
            name = "Unknown"
            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, face_enc)
                best = dists.argmin()
                if dists[best] < 0.6:
                    name = known_names[best]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def demographics_webcam() -> None:
    """Show age, gender and ethnicity predictions for the webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Nao foi possivel acessar a webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label = ""
        try:
            dem = detect_demographics(frame)
            parts = []
            gender = dem.get("gender")
            age = dem.get("age")
            ethnicity = dem.get("ethnicity")
            skin = dem.get("skin")
            if gender:
                parts.append(gender)
            if age:
                parts.append(age)
            if ethnicity:
                parts.append(ethnicity)
            if skin:
                parts.append(skin)
            label = ", ".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.error("demographics error: %s", exc)
        if label:
            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Reconhecimento facial")
    parser.add_argument("--webcam", action="store_true", help="Usa webcam")
    parser.add_argument("--image", help="Imagem para reconhecer", nargs="?")
    args = parser.parse_args()

    if args.webcam:
        recognize_webcam()
    elif args.image:
        print(recognize_faces(args.image))
