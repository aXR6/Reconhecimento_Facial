from typing import List

from transformers import pipeline

_pipe = None


def _load_pipe() -> None:
    """Inicializa o pipeline de legenda se ainda não estiver carregado."""
    global _pipe
    if _pipe is None:
        _pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def generate_caption(image_path: str) -> str:
    """Gera legenda da imagem usando modelo local da Hugging Face."""
    _load_pipe()
    if _pipe is None:
        raise RuntimeError("Falha ao carregar modelo de legenda")
    out: List[dict] = _pipe(image_path)
    if out:
        return out[0].get("generated_text", "")
    return ""
