import logging

logger = logging.getLogger(__name__)


def detect_demographics(image: str) -> dict:
    """Return gender, age, ethnicity and skin color labels from the image."""
    try:
        from .facexformer.inference import detect_demographics as fx_detect
        return fx_detect(image)
    except Exception as exc:  # noqa: BLE001
        logger.error("error detecting demographics: %s", exc)
        return {}
