import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from face_detection import detect_faces
from llm_service import generate_caption

logger = logging.getLogger(__name__)


def menu() -> None:
    while True:
        print("\n=== Menu ===")
        print("1 - Detectar rostos (OpenCV)")
        print("2 - Detectar rostos com MediaPipe (HuggingFace)")
        print("3 - Detectar rostos com YOLOv8 (HuggingFace)")
        print("4 - Gerar legenda via LLM")
        print("0 - Sair")
        choice = input("Escolha uma opcao: ").strip()

        if choice == "1":
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=False)
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == "2":
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=True, hf_model="mediapipe")
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == "3":
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=True, hf_model="yolov8")
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == "4":
            image = input("Caminho da imagem: ").strip()
            try:
                caption = generate_caption(image)
                print(f"Legenda gerada: {caption}")
            except Exception as exc:
                print(f"Erro ao gerar legenda: {exc}")
        elif choice == "0":
            break
        else:
            print("Opcao invalida")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ferramentas de reconhecimento facial"
    )
    sub = parser.add_subparsers(dest="cmd")

    detect_p = sub.add_parser("detect", help="Detectar rostos")
    detect_p.add_argument("--image", required=True)
    detect_p.add_argument("--output", default="output.jpg")
    detect_p.add_argument(
        "--model",
        choices=["opencv", "mediapipe", "yolov8"],
        default="opencv",
    )

    cap_p = sub.add_parser("caption", help="Gerar legenda")
    cap_p.add_argument("--image", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd is None:
        menu()
        return

    if args.cmd == "detect":
        use_hf = args.model in ("mediapipe", "yolov8")
        model = args.model if use_hf else "mediapipe"
        try:
            qtd = detect_faces(args.image, args.output, use_hf=use_hf, hf_model=model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Erro ao detectar rostos: %s", exc)
            return
        logger.info("Detectado(s) %s rosto(s). Resultado salvo em %s", qtd, args.output)
    elif args.cmd == "caption":
        try:
            caption = generate_caption(args.image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Erro ao gerar legenda: %s", exc)
            return
        print(caption)


if __name__ == "__main__":
    main()
