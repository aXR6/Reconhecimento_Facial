from typing import List

import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None

logger = logging.getLogger(__name__)
_pipe = None


def _load_pipe() -> None:
    """Inicializa o pipeline de legenda se ainda não estiver carregado."""
    global _pipe
    if _pipe is None:
        if pipeline is None:
            logger.error("transformers não está instalado")
            return
        model_name = os.getenv("HF_CAPTION_MODEL", "nlpconnect/vit-gpt2-image-captioning")
        try:
            from .device import torch_device

            device_str = torch_device()
            device = 0 if device_str == "cuda" else -1
            _pipe = pipeline(
                "image-to-text",
                model=model_name,
                device=device,
                use_fast=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao carregar modelo de legenda: %s", exc)


def generate_caption(image_path: str) -> str:
    """Gera legenda da imagem usando modelo local da Hugging Face."""
    _load_pipe()
    if _pipe is None:
        raise RuntimeError("Falha ao carregar modelo de legenda")
    try:
        out: List[dict] = _pipe(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro ao gerar legenda: %s", exc)
        return ""
    if out:
        return out[0].get("generated_text", "")
    return ""
