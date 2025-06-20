import argparse
import os
from typing import Optional

import cv2
import requests


def detect_faces(
    image_path: str,
    output_path: str = "output.jpg",
    use_hf: bool = False,
) -> int:
    """Detecta rostos em ``image_path`` e salva resultado em ``output_path``.

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
        url = os.getenv(
            "HF_FACE_DETECTION_URL",
            "https://api-inference.huggingface.co/models/qualcomm/MediaPipe-Face-Detection",
        )
        token = os.getenv("HUGGINGFACE_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with open(image_path, "rb") as f:
            data = f.read()
        resp = requests.post(url, headers=headers, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"Erro {resp.status_code}: {resp.text}")
        out = resp.json()
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"API error: {out['error']}")
        for det in out:
            box = det.get("box")
            if box:
                x1, y1 = int(box.get("xmin", 0)), int(box.get("ymin", 0))
                x2, y2 = int(box.get("xmax", 0)), int(box.get("ymax", 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
        help="Utiliza o modelo MediaPipe-Face-Detection da Hugging Face",
    )
    args = parser.parse_args()

    try:
        qtd = detect_faces(args.image, args.output, use_hf=args.hf)
    except FileNotFoundError as exc:
        print(exc)
        return

    print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {args.output}")


if __name__ == "__main__":
    main()
