import argparse
import logging
import queue
import threading
from typing import Optional

import numpy as np

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


def translate_microphone(
    model_name: str = "base",
    chunk_seconds: int = 5,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Capture audio from the microphone and translate chunks to English.

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
                        result = model.transcribe(audio, task="translate", language="pt", fp16=False)
                        text = result.get("text", "").strip()
                        if text:
                            print(text)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Erro na transcri\u00e7\u00e3o: %s", exc)
                    buffer = np.empty((0, 1), dtype=np.float32)
    except KeyboardInterrupt:
        print("Finalizado")


def translate_webcam(
    model_name: str = "base", chunk_seconds: int = 5
) -> None:
    """Translate microphone input while showing the webcam feed."""
    from .recognition import recognize_webcam

    stop_event = threading.Event()
    thr = threading.Thread(
        target=translate_microphone,
        args=(model_name, chunk_seconds, stop_event),
        daemon=True,
    )
    thr.start()
    try:
        recognize_webcam()
    finally:
        stop_event.set()
        thr.join()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Traduz audio do microfone em tempo real")
    parser.add_argument("--model", default="base", help="Modelo Whisper a ser usado")
    parser.add_argument("--chunk", type=int, default=5, help="Dura\u00e7\u00e3o de cada captura em segundos")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Abre a webcam enquanto traduz o áudio",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.webcam:
        translate_webcam(args.model, args.chunk)
    else:
        translate_microphone(args.model, args.chunk)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
