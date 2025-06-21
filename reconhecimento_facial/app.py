import argparse
import logging
import os
import time
import threading

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"

from dotenv import load_dotenv
import questionary

load_dotenv()

from reconhecimento_facial.face_detection import detect_faces
from reconhecimento_facial.llm_service import generate_caption
from reconhecimento_facial.obstruction_detection import detect_obstruction
from reconhecimento_facial.recognition import (
    recognize_webcam,
    register_person_webcam,
    demographics_webcam,
    recognize_webcam_mediapipe,
)
from reconhecimento_facial.preload import preload_models
from reconhecimento_facial.whisper_translation import (
    DEFAULT_WHISPER_MODEL,
    translate_microphone,
)

logger = logging.getLogger(__name__)

_src_lang = "pt"


def _language_menu() -> None:
    """Allow user to choose the source language for translation."""
    global _src_lang
    langs = {
        "Português": "pt",
        "English": "en",
        "Español": "es",
        "Français": "fr",
    }
    names = list(langs.keys())
    src = questionary.select("Idioma de entrada", choices=names).ask()
    if src:
        _src_lang = langs[src]


def _run_with_translation(func) -> None:
    """Execute a recognition function while running translation."""
    stop_event = threading.Event()
    thr = threading.Thread(
        target=translate_microphone,
        args=(DEFAULT_WHISPER_MODEL, 5, stop_event, _src_lang, True),
        daemon=True,
    )
    thr.start()
    try:
        func()
    finally:
        stop_event.set()
        thr.join()


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
        "Reconhecimento via webcam (face_recognition)",
        "Reconhecimento via webcam (MediaPipe)",
        "Demografia via webcam (FaceXFormer)",
        "Voltar",
    ]
    while True:
        choice = questionary.select("Reconhecimento", choices=options).ask()
        if choice in (None, "Voltar"):
            break
        _language_menu()
        if choice == options[0]:
            _run_with_translation(recognize_webcam)
        elif choice == options[1]:
            _run_with_translation(recognize_webcam_mediapipe)
        elif choice == options[2]:
            _run_with_translation(demographics_webcam)




def _device_menu() -> None:
    options = ["Auto", "GPU", "CPU", "Voltar"]
    while True:
        choice = questionary.select("Dispositivo de processamento", choices=options).ask()
        if choice in (None, "Voltar"):
            break
        from reconhecimento_facial.device import set_device

        if choice == "GPU":
            set_device("gpu")
        elif choice == "CPU":
            set_device("cpu")
        else:
            set_device("auto")
        print(f"Dispositivo selecionado: {choice}")


def _other_menu() -> None:
    options = [
        "Gerar legenda via LLM",
        "Selecionar dispositivo",
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
        elif choice == options[1]:
            _device_menu()


def _download_models() -> None:
    """Baixa todos os modelos utilizados pela aplicação."""
    print("Baixando modelos. Aguarde...")
    try:
        preload_models()
        print("Modelos baixados com sucesso")
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao baixar modelos: {exc}")


def menu() -> None:
    main_opts = [
        "Baixar modelos",
        "Detecção",
        "Reconhecimento",
        "Cadastrar pessoa (face_recognition)",
        "Outros",
        "Sair",
    ]
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        choice = questionary.select("Escolha uma categoria", choices=main_opts).ask()
        if choice in (None, "Sair"):
            break

        if choice == main_opts[0]:
            _download_models()
        elif choice == main_opts[1]:
            _detection_menu()
        elif choice == main_opts[2]:
            _recognition_menu()
        elif choice == main_opts[3]:
            name = input("Nome da pessoa: ").strip()
            try:
                if register_person_webcam(name):
                    time.sleep(2)
                else:
                    print("Erro ao cadastrar pessoa")
            except Exception as exc:
                print(f"Erro ao cadastrar: {exc}")
        elif choice == main_opts[4]:
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
