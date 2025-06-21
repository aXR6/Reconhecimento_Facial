import os
from typing import Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None

_DEVICE = os.getenv("RF_DEVICE", "auto").lower()


def set_device(device: str) -> None:
    """Define the processamento device ("cpu", "gpu" or "auto")."""
    global _DEVICE
    _DEVICE = device.lower()
    os.environ["RF_DEVICE"] = _DEVICE


def get_device() -> str:
    """Return the preferred device for inference."""
    if _DEVICE == "cpu":
        return "cpu"
    if _DEVICE == "gpu":
        return "cuda" if torch and getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch and getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"


def torch_device() -> str:
    """Return device string compatible with torch."""
    dev = get_device()
    return "cuda" if dev == "cuda" else "cpu"

