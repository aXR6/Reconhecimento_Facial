# Helper to preload models for the interactive menu
from __future__ import annotations

from reconhecimento_facial.llm_service import _load_pipe as _load_llm
from reconhecimento_facial.obstruction_detection import _load_pipe as _load_obstruction
from reconhecimento_facial.facexformer.inference import _load_model as _load_facexformer


def preload_models() -> None:
    """Carrega todos os modelos utilizados pela aplicacao."""
    try:
        _load_llm()
    except Exception:
        pass

    try:
        _load_obstruction()
    except Exception:
        pass

    try:
        _load_facexformer()
    except Exception:
        pass
