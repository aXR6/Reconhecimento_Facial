import logging
import numpy as np
import sys
import types
import threading
import tempfile
from pathlib import Path
from typing import Iterable, Any
from importlib.util import find_spec
from datetime import datetime

import cv2
import os

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "photos")
Path(PHOTOS_DIR).mkdir(parents=True, exist_ok=True)

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
    module.cnn_face_detector_model_location = lambda: _resource(
        "mmod_human_face_detector.dat"
    )

    sys.modules["face_recognition_models"] = module


_patch_face_recognition_models()

try:
    import face_recognition
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    face_recognition = None

from reconhecimento_facial.db import get_conn, init_db
from reconhecimento_facial.demographics_detection import detect_demographics
from reconhecimento_facial.facexformer import analyze_face
from reconhecimento_facial.google_search import run_google_search


def _crop_and_save_face(image_path: str) -> None:
    """Recorta o primeiro rosto encontrado e salva em ``PHOTOS_DIR``."""
    img = cv2.imread(image_path)
    if img is None:
        return
    crop = None
    if face_recognition is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if locs:
            top, right, bottom, left = locs[0]
            crop = img[top:bottom, left:right]
    if crop is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop = img[y : y + h, x : x + w]
    if crop is not None:
        Path(PHOTOS_DIR).mkdir(parents=True, exist_ok=True)
        out_path = Path(PHOTOS_DIR) / Path(image_path).name
        cv2.imwrite(str(out_path), crop)
        cv2.imwrite(image_path, crop)


def _save_cropped_face(crop: np.ndarray, name: str, tech: str) -> Path:
    """Save cropped face image using standardized naming."""
    timestamp = datetime.now().strftime("%d_%m_%Y")
    clean_name = name.replace(" ", "_") or "Unknown"
    filename = f"{clean_name}_{timestamp}_{tech}.jpg"
    out_path = Path(PHOTOS_DIR) / filename
    Path(PHOTOS_DIR).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return out_path


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
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("c"), ord(" ")):
            cv2.imwrite(tmp_path, frame)
            _crop_and_save_face(tmp_path)
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


def register_person_webcam(
    name: str,
    *,
    google_search: bool = False,
) -> bool:
    """Capture an image from the webcam and register the person.

    When ``google_search`` is ``True``, the captured face is also searched on
    Google in the background. The function returns ``True``
    only when both the capture and the database insertion succeed. This
    behaviour ensures that callers do not display a successful message when the
    registration actually failed.
    """
    tmp = f"/tmp/{name.replace(' ', '_')}.jpg"
    if not capture_from_webcam(tmp):
        return False
    try:
        ok = register_person(name, tmp)
        if ok:
            if google_search:
                thr = threading.Thread(
                    target=_google_search_background,
                    args=(tmp,),
                    daemon=True,
                )
                thr.start()
            print("Cadastro salvo com sucesso")
        return ok
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _google_search_background(img_path: str) -> None:
    """Run Google search for ``img_path`` and clean up."""
    try:
        run_google_search([img_path])
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("google search error: %s", exc)
    finally:
        try:
            if img_path.startswith("/tmp/"):
                os.remove(img_path)
        except OSError:
            pass


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


def recognize_faces_google(image_path: str) -> list[str]:
    """Recognize faces and search the image on Google in the background."""
    names = recognize_faces(image_path)
    if names:
        thr = threading.Thread(
            target=_google_search_background,
            args=(image_path,),
            daemon=True,
        )
        thr.start()
    return names


def recognize_webcam(*, google_search: bool = False) -> None:
    """Capture webcam and display faces in real time.

    When ``google_search`` is True, recognized faces are searched on Google in
    the background.
    """
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

    seen: set[str] = set()

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

            _save_cropped_face(crop, name, "face_recognition")
            if google_search and name != "Unknown" and name not in seen:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, crop)
                thr = threading.Thread(
                    target=_google_search_background,
                    args=(tmp.name,),
                    daemon=True,
                )
                thr.start()
                seen.add(name)
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def recognize_webcam_mediapipe(*, google_search: bool = False) -> None:
    """Capture webcam using MediaPipe for detection and identify faces.

    When ``google_search`` is True, the cropped face is searched on Google in
    the background.
    """
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

    seen: set[str] = set()

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
            _save_cropped_face(crop, name, "mediapipe")
            if google_search and name != "Unknown" and name not in seen:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, crop)
                thr = threading.Thread(
                    target=_google_search_background,
                    args=(tmp.name,),
                    daemon=True,
                )
                thr.start()
                seen.add(name)
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def demographics_webcam(*, google_search: bool = False) -> None:
    """Recognize people and display FaceXFormer predictions using the webcam feed."""
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

    seen: set[str] = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Nao foi possivel acessar a webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for idx, ((top, right, bottom, left), face_enc) in enumerate(
            zip(locations, encodings)
        ):
            name = "Unknown"
            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, face_enc)
                best = dists.argmin()
                if dists[best] < 0.6:
                    name = known_names[best]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            crop = frame[top:bottom, left:right]

            _save_cropped_face(crop, name, "facexformer")
            if google_search and name != "Unknown" and name not in seen:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, crop)
                thr = threading.Thread(
                    target=_google_search_background,
                    args=(tmp.name,),
                    daemon=True,
                )
                thr.start()
                seen.add(name)

            dem: dict[str, Any] = {}
            try:
                dem = analyze_face(crop)
            except Exception as exc:  # noqa: BLE001
                logger.error("demographics error: %s", exc)

            parts = [name]
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
            info = ", ".join(parts)
            cv2.putText(
                frame,
                info,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if idx == 0:
                y = bottom + 20
                hp = dem.get("headpose")
                if hp:
                    pose = f"pitch:{hp['pitch']:.1f} yaw:{hp['yaw']:.1f} roll:{hp['roll']:.1f}"
                    cv2.putText(
                        frame,
                        pose,
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    y += 20

                attrs = dem.get("attributes")
                if attrs:
                    positives = [k for k, v in attrs.items() if v]
                    if positives:
                        cv2.putText(
                            frame,
                            ", ".join(positives[:5]),
                            (left, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                        y += 20

                vis = dem.get("visibility")
                if vis is not None:
                    cv2.putText(
                        frame,
                        f"visibility: {vis}",
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                seg = dem.get("segmentation")
                if seg is not None:
                    seg_img = np.array(seg, dtype=np.uint8)
                    seg_img = cv2.applyColorMap(
                        (seg_img * 20).astype(np.uint8), cv2.COLORMAP_JET
                    )
                    cv2.imshow("segmentation", seg_img)

                landmarks = dem.get("landmarks")
                if landmarks:
                    lm_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    for x, z in landmarks:
                        cv2.circle(lm_img, (int(x), int(z)), 1, (0, 255, 0), -1)
                    cv2.imshow("landmarks", lm_img)

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
    parser.add_argument("--google-search", action="store_true", help="Buscar no Google")
    args = parser.parse_args()

    if args.webcam:
        recognize_webcam(google_search=args.google_search)
    elif args.image:
        if args.google_search:
            print(recognize_faces_google(args.image))
        else:
            print(recognize_faces(args.image))
