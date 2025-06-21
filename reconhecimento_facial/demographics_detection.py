import logging
import os

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None

logger = logging.getLogger(__name__)
_pipe = None


def _load_pipe() -> None:
    """Load demographics classification pipeline."""
    global _pipe
    if _pipe is None:
        if pipeline is None:
            logger.error("transformers not installed")
            return
        model = os.getenv(
            "DEMOGRAPHICS_MODEL_REPO", "nateraw/age-gender-estimation"
        )
        try:
            if model == "kartiknarayan/facexformer":
                _pipe = "facexformer"
            else:
                _pipe = pipeline("image-classification", model=model)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load demographics model: %s", exc)


def detect_demographics(image) -> dict:
    """Return gender, age, ethnicity and skin color labels from the image."""
    _load_pipe()
    if _pipe is None:
        raise RuntimeError("Demographics model not available")
    if _pipe == "facexformer":
        try:
            from .facexformer.inference import detect_demographics as fx_detect
            return fx_detect(image)
        except Exception as exc:  # noqa: BLE001
            logger.error("error detecting facexformer demographics: %s", exc)
            return {}
    try:
        preds = _pipe(image)
    except Exception as exc:  # noqa: BLE001
        logger.error("error detecting demographics: %s", exc)
        return {}
    result = {}
    for pred in preds:
        label = pred.get("label", "")
        if not label:
            continue
        if ":" in label:
            key, value = [p.strip() for p in label.split(":", 1)]
            result[key.lower()] = value
    return result
