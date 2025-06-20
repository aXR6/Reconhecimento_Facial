import argparse
import logging
import os
import time

from dotenv import load_dotenv
import questionary

load_dotenv()

from face_detection import detect_faces
from llm_service import generate_caption
from obstruction_detection import detect_obstruction
from recognition import recognize_webcam, register_person_webcam

logger = logging.getLogger(__name__)


def _detection_menu() -> None:
    options = [
        "Detectar rostos (OpenCV)",
        "Detectar rostos com MediaPipe",
        "Detectar rostos com YOLOv8",
        "Detectar obstru\u00e7\u00e3o facial",
        "Voltar",
    ]
    while True:
        choice = questionary.select("Detec\u00e7\u00e3o", choices=options).ask()
        if choice in (None, "Voltar"):
            break

        if choice == options[0]:
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=False)
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == options[1]:
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=True, hf_model="mediapipe")
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == options[2]:
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            try:
                qtd = detect_faces(image, output, use_hf=True, hf_model="yolov8")
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == options[3]:
            image = input("Caminho da imagem: ").strip()
            try:
                label = detect_obstruction(image)
                print(f"Obstru\u00e7\u00e3o detectada: {label}")
            except Exception as exc:
                print(f"Erro: {exc}")


def _recognition_menu() -> None:
    options = [
        "Cadastrar pessoa",
        "Reconhecimento via webcam",
        "Voltar",
    ]
    while True:
        choice = questionary.select("Reconhecimento", choices=options).ask()
        if choice in (None, "Voltar"):
            break

        if choice == options[0]:
            name = input("Nome da pessoa: ").strip()
            try:
                if register_person_webcam(name):
                    time.sleep(2)
                else:
                    print("Erro ao cadastrar pessoa")
            except Exception as exc:
                print(f"Erro ao cadastrar: {exc}")
        elif choice == options[1]:
            recognize_webcam()


def _other_menu() -> None:
    options = [
        "Gerar legenda via LLM",
        "Voltar",
    ]
    while True:
        choice = questionary.select("Outros", choices=options).ask()
        if choice in (None, "Voltar"):
            break

        if choice == options[0]:
            image = input("Caminho da imagem: ").strip()
            try:
                caption = generate_caption(image)
                print(f"Legenda gerada: {caption}")
            except Exception as exc:
                print(f"Erro ao gerar legenda: {exc}")


def menu() -> None:
    main_opts = ["Detec\u00e7\u00e3o", "Reconhecimento", "Outros", "Sair"]
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        choice = questionary.select("Escolha uma categoria", choices=main_opts).ask()
        if choice in (None, "Sair"):
            break

        if choice == main_opts[0]:
            _detection_menu()
        elif choice == main_opts[1]:
            _recognition_menu()
        elif choice == main_opts[2]:
            _other_menu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferramentas de reconhecimento facial")
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

    obs_p = sub.add_parser("obstruction", help="Detectar obstru\u00e7\u00e3o facial")
    obs_p.add_argument("--image", required=True)

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
    elif args.cmd == "obstruction":
        try:
            label = detect_obstruction(args.image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Erro ao detectar obstru\u00e7\u00e3o: %s", exc)
            return
        print(label)


if __name__ == "__main__":
    main()
