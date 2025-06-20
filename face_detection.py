import argparse
from typing import Optional

import cv2
from huggingface_hub import hf_hub_download
import mediapipe as mp
from ultralytics import YOLO


def detect_faces(
    image_path: str,
    output_path: str = "output.jpg",
    use_hf: bool = False,
    hf_model: str = "mediapipe",
) -> int:
    """Detecta rostos em ``image_path`` e salva resultado em ``output_path``.

    Se ``use_hf`` for ``True``, utiliza modelos da Hugging Face localmente
    (``mediapipe`` ou ``yolov8``) para complementar a detecção.

    Retorna o número de rostos encontrados.
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

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if use_hf:
        if hf_model == "yolov8":
            weight = hf_hub_download(
                "jaredthejelly/yolov8s-face-detection", "YOLOv8-face-detection.pt"
            )
            yolo = YOLO(weight)
            results = yolo(img, verbose=False)[0]
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
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

    cv2.imwrite(output_path, img)
    return len(faces)


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

    try:
        qtd = detect_faces(args.image, args.output, use_hf=args.hf, hf_model=args.model)
    except FileNotFoundError as exc:
        print(exc)
        return

    print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {args.output}")


if __name__ == "__main__":
    main()
