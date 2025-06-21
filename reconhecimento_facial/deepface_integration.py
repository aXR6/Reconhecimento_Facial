import logging

try:
    from deepface import DeepFace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DeepFace = None

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tf = None

logger = logging.getLogger(__name__)
_model_loaded = False


def _load_model() -> None:
    """Preload DeepFace models."""
    global _model_loaded
    if _model_loaded or DeepFace is None:
        return
    try:
        from .device import get_device

        if tf is not None and get_device() == "cpu":
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass

        DeepFace.build_model("Age")
        DeepFace.build_model("Gender")
        DeepFace.build_model("Race")
        _model_loaded = True
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load DeepFace models: %s", exc)


def detect_demographics(image_path: str) -> dict:
    """Detect age, gender and ethnicity using DeepFace."""
    _load_model()
    if DeepFace is None:
        raise RuntimeError("DeepFace not installed")
    from .device import get_device
    if tf is not None and get_device() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    try:
        res = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender", "race"],
            enforce_detection=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to analyze face with DeepFace: %s", exc)
        raise
    if isinstance(res, list):
        res = res[0]
    return {
        "age": str(res.get("age", "")),
        "gender": str(res.get("gender", "")).lower(),
        "ethnicity": str(res.get("dominant_race", "")),
        "skin": str(res.get("dominant_race", "")),
    }
