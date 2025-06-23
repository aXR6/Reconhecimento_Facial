"""Minimal FaceXFormer integration for demographics detection."""

from .inference import analyze_face, detect_demographics, extract_embedding

__all__ = ["detect_demographics", "analyze_face", "extract_embedding"]
