import os

from dotenv import load_dotenv
import requests

load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"


def generate_caption(image_path: str, token: str | None = None) -> str:
    """Envia a imagem para a API de inferência da Hugging Face e retorna a legenda.

    O ``token`` é opcional e pode ser utilizado para autenticação na API.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(image_path, "rb") as f:
        data = f.read()

    resp = requests.post(API_URL, headers=headers, data=data)
    if resp.status_code != 200:
        raise RuntimeError(f"Erro {resp.status_code}: {resp.text}")

    out = resp.json()
    if isinstance(out, list) and out:
        return out[0].get("generated_text", "")
    if isinstance(out, dict):
        return out.get("generated_text", "")
    return str(out)
