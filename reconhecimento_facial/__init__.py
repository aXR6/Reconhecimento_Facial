"""Reconhecimento Facial package."""

from .device import get_device, set_device


def translate_microphone(*a, **kw):  # pragma: no cover - wrapper for lazy import
    from .whisper_translation import translate_microphone as _tm

    return _tm(*a, **kw)


def translate_file(*a, **kw):  # pragma: no cover - wrapper for lazy import
    from .whisper_translation import translate_file as _tf

    return _tf(*a, **kw)


def translate_webcam(*a, **kw):  # pragma: no cover - wrapper for lazy import
    from .whisper_translation import translate_webcam as _tw

    return _tw(*a, **kw)


__all__ = [
    "get_device",
    "set_device",
    "translate_microphone",
    "translate_file",
    "translate_webcam",
]
