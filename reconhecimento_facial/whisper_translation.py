import argparse
import logging
import os
import queue
import threading
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except (ModuleNotFoundError, OSError):  # pragma: no cover - optional dependency
    sd = None

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline as hf_pipeline,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hf_pipeline = None

logger = logging.getLogger(__name__)

# Default Whisper model used when none is provided
DEFAULT_WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL", "openai/whisper-large-v3-turbo"
)


def _get_pipe(model_name: str):
    """Return a Hugging Face ASR pipeline for ``model_name``."""
    if hf_pipeline is None:
        raise ImportError("transformers não instaladas")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    # Avoid warning when passing ``task=translate`` by clearing
    # ``forced_decoder_ids`` from the generation config.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_decoder_ids = None
    processor = AutoProcessor.from_pretrained(model_name)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


def _process_file(
    file_path: str,
    model_name: str,
    task: str,
    source_lang: Optional[str],
) -> str:
    """Run Whisper on ``file_path`` with the given task."""
    if hf_pipeline is None:
        logger.error("Dependências não instaladas: transformers")
        return ""
    try:
        pipe = _get_pipe(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao carregar modelo Whisper: %s", exc)
        return ""
    gen_kwargs = {"task": task}
    if source_lang:
        gen_kwargs["language"] = source_lang
    try:
        result = pipe(file_path, generate_kwargs=gen_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro na transcrição: %s", exc)
        return ""
    return result.get("text", "").strip()


def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate ``text`` from ``source_lang`` to ``target_lang`` using a
    Hugging Face model."""

    if hf_pipeline is None:
        logger.error("Depend\u00eancias n\u00e3o instaladas: transformers")
        return ""

    model_name = os.getenv("TRANSLATION_MODEL", "facebook/m2m100_418M")
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt")
        forced_id = tokenizer.get_lang_id(target_lang)
        outputs = model.generate(**encoded, forced_bos_token_id=forced_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro na tradu\u00e7\u00e3o: %s", exc)
        return ""


def translate_file(
    file_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    source_lang: Optional[str] = None,
    target_lang: str = "en",
) -> str:
    """Translate an audio file using Whisper and an optional translation model."""

    if target_lang == "en":
        text = _process_file(file_path, model_name, "translate", source_lang)
        if text:
            return text

    text = _process_file(file_path, model_name, "transcribe", source_lang)
    if not text:
        return ""
    return _translate_text(text, source_lang or "en", target_lang)


def transcribe_file(
    file_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    source_lang: Optional[str] = None,
) -> str:
    """Transcribe an audio file using Whisper."""
    return _process_file(file_path, model_name, "transcribe", source_lang)


def whisper_translate_file(
    file_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    source_lang: Optional[str] = None,
    target_lang: str = "en",
) -> str:
    """Compatibility alias for :func:`translate_file`."""
    return translate_file(file_path, model_name, source_lang, target_lang)




def translate_microphone(
    model_name: str = DEFAULT_WHISPER_MODEL,
    chunk_seconds: int = 5,
    stop_event: Optional[threading.Event] = None,
    source_lang: str = "pt",
    target_lang: str = "en",
    translate: bool = True,
) -> None:
    """Capture audio and optionally translate chunks using Whisper.

    Parameters
    ----------
    model_name:
        Whisper model to use.
    chunk_seconds:
        Duration of each audio chunk to be transcribed.
    stop_event:
        Optional event to signal when the loop should stop. Useful when running
        the translation in a background thread.
    translate:
        When ``True`` translate the audio. Otherwise just transcribe.
    """
    if hf_pipeline is None or sd is None:
        logger.error("Depend\u00eancias n\u00e3o instaladas: transformers ou sounddevice")
        return
    try:
        pipe = _get_pipe(model_name)
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
                        task = "translate" if translate and target_lang == "en" else "transcribe"
                        gen_kwargs = {"task": task, "language": source_lang}
                        result = pipe(audio, generate_kwargs=gen_kwargs)
                        text = result.get("text", "").strip()
                        if text:
                            if translate and target_lang != "en":
                                text = _translate_text(text, source_lang, target_lang)
                            if text:
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
    translate: bool = True,
) -> None:
    """Translate microphone input while showing the webcam feed.

    Parameters
    ----------
    translate:
        When ``True`` translate the audio. Otherwise just transcribe.
    """
    if recognition_func is None:
        from .recognition import recognize_webcam as recognition_func

    stop_event = threading.Event()
    thr = threading.Thread(
        target=translate_microphone,
        args=(model_name, chunk_seconds, stop_event, source_lang, target_lang, translate),
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
    parser.add_argument(
        "--dst",
        default="en",
        help="Idioma de saída",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Apenas transcreve o áudio em vez de traduzir",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Abre a webcam enquanto traduz o áudio",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.file:
        if args.transcribe:
            text = transcribe_file(args.file, args.model, args.src)
        else:
            text = translate_file(args.file, args.model, args.src, args.dst)
        if text:
            print(text)
        if args.expected is not None:
            if text.strip().lower() == args.expected.strip().lower():
                print("Tradução confere")
            else:
                print("Tradução diferente do esperado")
    elif args.webcam:
        translate_webcam(
            args.model,
            args.chunk,
            args.src,
            args.dst,
            translate=not args.transcribe,
        )
    else:
        translate_microphone(
            args.model,
            args.chunk,
            None,
            args.src,
            args.dst,
            translate=not args.transcribe,
        )


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
