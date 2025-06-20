import argparse
import logging
import os
from typing import Optional

import cv2
try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hf_hub_download = None

try:
    import mediapipe as mp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mp = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    YOLO = None

logger = logging.getLogger(__name__)


def detect_faces(
    image_path: str,
    output_path: str = "output.jpg",
    use_hf: bool = False,
    hf_model: str = "mediapipe",
) -> int:
    """Detecta rostos em ``image_path`` e salva resultado em ``output_path``.

    Se ``use_hf`` for ``True``, utiliza modelos da Hugging Face localmente
    (``mediapipe`` ou ``yolov8``) para complementar a detecção.

    Retorna o número total de rostos encontrados, somando as
    detecções do Haar Cascade e dos modelos da Hugging Face
    quando ``use_hf`` for ``True``.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    total_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if use_hf:
        if hf_model == "yolov8":
            repo = os.getenv("YOLOV8_REPO", "jaredthejelly/yolov8s-face-detection")
            try:
                if hf_hub_download is None or YOLO is None:
                    raise RuntimeError("Dependências do YOLOv8 não instaladas")
                weight = hf_hub_download(repo, "YOLOv8-face-detection.pt")
                yolo = YOLO(weight)
                results = yolo(img, verbose=False)[0]
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                total_faces += len(results.boxes)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao usar modelo YOLOv8: %s", exc)
        else:
            repo = os.getenv("MEDIAPIPE_REPO", "qualcomm/MediaPipe-Face-Detection")
            try:
                if mp is None:
                    raise RuntimeError("Dependências do MediaPipe não instaladas")
                mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)
                if res.detections:
                    for det in res.detections:
                        box = det.location_data.relative_bounding_box
                        x = int(box.xmin * img.shape[1])
                        y = int(box.ymin * img.shape[0])
                        w = int(box.width * img.shape[1])
                        h = int(box.height * img.shape[0])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    total_faces += len(res.detections)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao usar modelo MediaPipe: %s", exc)

    cv2.imwrite(output_path, img)
    return total_faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta rostos em uma imagem.")
    parser.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    parser.add_argument(
        "--output", default="output.jpg", help="Arquivo de saída com detecções"
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Utiliza modelo da Hugging Face (ver --model)",
    )
    parser.add_argument(
        "--model",
        choices=["mediapipe", "yolov8"],
        default="mediapipe",
        help="Modelo a ser usado com --hf",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        qtd = detect_faces(args.image, args.output, use_hf=args.hf, hf_model=args.model)
    except FileNotFoundError as exc:
        logger.error(exc)
        return

    logger.info("Detectado(s) %s rosto(s). Resultado salvo em %s", qtd, args.output)


if __name__ == "__main__":
    main()
