"""Reconhecimento Facial package."""

from .device import get_device, set_device

def translate_microphone(*a, **kw):  # pragma: no cover - wrapper for lazy import
    from .whisper_translation import translate_microphone as _tm

    return _tm(*a, **kw)

__all__ = ["get_device", "set_device", "translate_microphone"]
