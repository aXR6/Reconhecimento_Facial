import os
from typing import Optional

from face_detection import detect_faces
from llm_service import generate_caption


def menu() -> None:
    while True:
        print("\n=== Menu ===")
        print("1 - Detectar rostos em imagem")
        print("2 - Gerar legenda via LLM (HuggingFace)")
        print("0 - Sair")
        choice = input("Escolha uma opcao: ").strip()

        if choice == "1":
            image = input("Caminho da imagem: ").strip()
            output = input("Arquivo de saida [output.jpg]: ").strip() or "output.jpg"
            use_hf = input("Usar modelo da HuggingFace? [n]: ").strip().lower() == "s"
            try:
                qtd = detect_faces(image, output, use_hf=use_hf)
                print(f"Detectado(s) {qtd} rosto(s). Resultado salvo em {output}")
            except Exception as exc:
                print(f"Erro: {exc}")
        elif choice == "2":
            image = input("Caminho da imagem: ").strip()
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                print("Defina a variavel de ambiente HUGGINGFACE_TOKEN com seu token.")
                continue
            try:
                caption = generate_caption(image, token)
                print(f"Legenda gerada: {caption}")
            except Exception as exc:
                print(f"Erro ao gerar legenda: {exc}")
        elif choice == "0":
            break
        else:
            print("Opcao invalida")


if __name__ == "__main__":
    menu()
