import argparse
import logging
import os
import queue
import threading
from typing import Optional

import numpy as np

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pipeline = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

try:
    import sounddevice as sd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sd = None

try:
    import whisper
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    whisper = None

logger = logging.getLogger(__name__)
_translation_pipes: dict[tuple[str, str], any] = {}

# Default Whisper model used when none is provided
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


def _load_translator(src_lang: str, tgt_lang: str):
    """Load translation pipeline for the given languages."""
    key = (src_lang, tgt_lang)
    pipe = _translation_pipes.get(key)
    if pipe is None:
        if pipeline is None:
            logger.error("transformers não está instalado")
            return None
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        try:
            pipe = pipeline("translation", model=model_name)
            _translation_pipes[key] = pipe
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao carregar modelo de tradução: %s", exc)
            return None
    return pipe


def _translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    pipe = _load_translator(src_lang, tgt_lang)
    if pipe is None:
        return text
    try:
        out = pipe(text)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro na tradução: %s", exc)
        return text
    return out[0].get("translation_text", text)


def translate_file(
    file_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    source_lang: str = "pt",
    target_lang: str = "en",
) -> str:
    """Translate an audio file using Whisper and a translation model.

    Parameters
    ----------
    file_path:
        Path to the audio file.
    model_name:
        Whisper model to use.

    Returns
    -------
    str
        Translated text. Empty string on failure.
    """
    if whisper is None:
        logger.error("Depend\u00eancias n\u00e3o instaladas: whisper")
        return ""
    try:
        model = whisper.load_model(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao carregar modelo Whisper: %s", exc)
        return ""
    try:
        result = model.transcribe(
            file_path,
            task="transcribe",
            language=source_lang,
            fp16=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro na transcri\u00e7\u00e3o: %s", exc)
        return ""
    text = result.get("text", "").strip()
    if not text:
        return ""
    if source_lang != target_lang:
        text = _translate_text(text, source_lang, target_lang)
    return text


def translate_microphone(
    model_name: str = DEFAULT_WHISPER_MODEL,
    chunk_seconds: int = 5,
    stop_event: Optional[threading.Event] = None,
    source_lang: str = "pt",
    target_lang: str = "en",
) -> None:
    """Capture audio and translate chunks from ``source_lang`` to ``target_lang``.

    Parameters
    ----------
    model_name:
        Whisper model to use.
    chunk_seconds:
        Duration of each audio chunk to be transcribed.
    stop_event:
        Optional event to signal when the loop should stop. Useful when running
        the translation in a background thread.
    """
    if whisper is None or sd is None:
        logger.error("Depend\u00eancias n\u00e3o instaladas: whisper ou sounddevice")
        return
    try:
        model = whisper.load_model(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao carregar modelo Whisper: %s", exc)
        return

    samplerate = 16000
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time, status):  # noqa: D401 - sounddevice callback
        if status:
            logger.warning(status)
        q.put(indata.copy())

    if stop_event is None:
        print("Pressione Ctrl+C para encerrar")
    buffer = np.empty((0, 1), dtype=np.float32)
    try:
        with sd.InputStream(callback=callback):
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                data = q.get()
                buffer = np.concatenate([buffer, data])
                if buffer.shape[0] / samplerate >= chunk_seconds:
                    audio = buffer[:, 0]
                    try:
                        result = model.transcribe(
                            audio,
                            task="transcribe",
                            language=source_lang,
                            fp16=False,
                        )
                        text = result.get("text", "").strip()
                        if text:
                            if source_lang != target_lang:
                                text = _translate_text(text, source_lang, target_lang)
                            print(text)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Erro na transcri\u00e7\u00e3o: %s", exc)
                    buffer = np.empty((0, 1), dtype=np.float32)
    except KeyboardInterrupt:
        print("Finalizado")


def translate_webcam(
    model_name: str = DEFAULT_WHISPER_MODEL,
    chunk_seconds: int = 5,
    source_lang: str = "pt",
    target_lang: str = "en",
    recognition_func=None,
) -> None:
    """Translate microphone input while showing the webcam feed."""
    if recognition_func is None:
        from .recognition import recognize_webcam as recognition_func

    stop_event = threading.Event()
    thr = threading.Thread(
        target=translate_microphone,
        args=(model_name, chunk_seconds, stop_event, source_lang, target_lang),
        daemon=True,
    )
    thr.start()
    try:
        recognition_func()
    finally:
        stop_event.set()
        thr.join()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Traduz áudio usando Whisper")
    parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        help="Modelo Whisper a ser usado",
    )
    parser.add_argument("--chunk", type=int, default=5, help="Duração de cada captura em segundos")
    parser.add_argument("--file", help="Arquivo de áudio a ser traduzido")
    parser.add_argument("--expected", help="Tradução esperada para o áudio")
    parser.add_argument("--src", default="pt", help="Idioma de entrada")
    parser.add_argument("--tgt", default="en", help="Idioma de saída")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Abre a webcam enquanto traduz o áudio",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.file:
        text = translate_file(args.file, args.model, args.src, args.tgt)
        if text:
            print(text)
        if args.expected is not None:
            if text.strip().lower() == args.expected.strip().lower():
                print("Tradução confere")
            else:
                print("Tradução diferente do esperado")
    elif args.webcam:
        translate_webcam(args.model, args.chunk, args.src, args.tgt)
    else:
        translate_microphone(args.model, args.chunk, None, args.src, args.tgt)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
