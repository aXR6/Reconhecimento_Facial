from typing import List

import logging
import os

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
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
        if all(hasattr(_pipe, attr) for attr in ("preprocess", "_forward", "postprocess")):
            model_inputs = _pipe.preprocess(image_path)
            if (
                "input_ids" in model_inputs
                and model_inputs["input_ids"] is not None
                and "attention_mask" not in model_inputs
            ):
                import torch  # Local import to avoid hard dependency during tests

                model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
            output_ids = _pipe._forward(model_inputs)
            out: List[dict] = _pipe.postprocess(output_ids)
        else:
            out: List[dict] = _pipe(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro ao gerar legenda: %s", exc)
        return ""
    caption = out[0].get("generated_text", "") if out else ""
    try:  # enrich caption with demographic information
        from .demographics_detection import detect_demographics

        info = detect_demographics(image_path)
        parts = []
        if isinstance(info, dict):
            gender = info.get("gender")
            age = info.get("age")
            ethnicity = info.get("ethnicity")
            skin = info.get("skin")
            if gender:
                parts.append(f"gender: {gender}")
            if age:
                parts.append(f"age: {age}")
            if ethnicity:
                parts.append(f"ethnicity: {ethnicity}")
            if skin:
                parts.append(f"skin: {skin}")
        if parts:
            caption = f"{caption} ({', '.join(parts)})"
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro ao enriquecer legenda: %s", exc)
    return caption
