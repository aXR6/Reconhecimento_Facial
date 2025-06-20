import argparse
import cv2


def detect_faces(image_path: str, output_path: str = "output.jpg") -> int:
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

    cv2.imwrite(output_path, img)
    return len(faces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta rostos em uma imagem.")
    parser.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    parser.add_argument(
        "--output", default="output.jpg", help="Arquivo de saída com detecções"
    )
    args = parser.parse_args()

    try:
        qtd = detect_faces(args.image, args.output)
    except FileNotFoundError as exc:
        print(exc)
        return

    print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {args.output}")


if __name__ == "__main__":
    main()
