import logging
import os

logger = logging.getLogger(__name__)
_backend = os.getenv("DEMOGRAPHICS_BACKEND", "facexformer")


def set_backend(backend: str) -> None:
    """Define which backend will be used for demographics detection."""
    global _backend
    _backend = backend


def detect_demographics(image: str) -> dict:
    """Return gender, age, ethnicity and skin color labels from the image."""
    try:
        if _backend == "deepface":
            from .deepface_integration import detect_demographics as dp_detect
            return dp_detect(image)
        from .facexformer.inference import detect_demographics as fx_detect
        return fx_detect(image)
    except Exception as exc:  # noqa: BLE001
        logger.error("error detecting demographics: %s", exc)
        return {}
