import logging
import os

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None

logger = logging.getLogger(__name__)
_pipe = None


def _load_pipe() -> None:
    """Load classification pipeline if not already loaded."""
    global _pipe
    if _pipe is None:
        if pipeline is None:
            logger.error("transformers n\u00e3o est\u00e1 instalado")
            return
        model = os.getenv(
            "OBSTRUCTION_MODEL_REPO",
            "dima806/face_obstruction_image_detection",
        )
        try:
            _pipe = pipeline("image-classification", model=model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao carregar modelo de obstru\u00e7\u00e3o: %s", exc)


def detect_obstruction(image_path: str) -> str:
    """Return detected obstruction label from the given image."""
    _load_pipe()
    if _pipe is None:
        raise RuntimeError("Falha ao carregar modelo de obstru\u00e7\u00e3o")
    try:
        preds = _pipe(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro ao classificar obstru\u00e7\u00e3o: %s", exc)
        return ""
    if preds:
        return preds[0].get("label", "")
    return ""
