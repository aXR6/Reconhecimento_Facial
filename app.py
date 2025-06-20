import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from face_detection import detect_faces
from llm_service import generate_caption


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


if __name__ == "__main__":
    menu()
