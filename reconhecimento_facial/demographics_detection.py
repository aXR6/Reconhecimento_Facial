import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_demographics(image: Any) -> dict:
    """Return gender, age, ethnicity and skin color labels from the image.

    Parameters
    ----------
    image:
        Path to the image or an array (BGR) representing the image.
    """
    try:
        from .facexformer.inference import detect_demographics as fx_detect
        return fx_detect(image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("error detecting demographics: %s", exc)
        return {}
