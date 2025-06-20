import logging
import os

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None

logger = logging.getLogger(__name__)
_pipe = None


def _load_pipe() -> None:
    global _pipe
    if _pipe is None:
        if pipeline is None:
            logger.error("transformers not installed")
            return
        model = os.getenv("EMOTION_MODEL_REPO", "nateraw/fer-vit-base")
        try:
            _pipe = pipeline("image-classification", model=model)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load emotion model: %s", exc)


def detect_emotion(image_path: str) -> str:
    _load_pipe()
    if _pipe is None:
        raise RuntimeError("Emotion model not available")
    try:
        res = _pipe(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("error detecting emotion: %s", exc)
        return ""
    if res:
        return res[0].get("label", "")
    return ""
