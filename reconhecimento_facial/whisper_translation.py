import argparse
import logging
import queue
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


def translate_microphone(model_name: str = "base", chunk_seconds: int = 5) -> None:
    """Capture audio from the microphone and translate chunks to English."""
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

    print("Pressione Ctrl+C para encerrar")
    buffer = np.empty((0, 1), dtype=np.float32)
    try:
        with sd.InputStream(callback=callback):
            while True:
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Traduz audio do microfone em tempo real")
    parser.add_argument("--model", default="base", help="Modelo Whisper a ser usado")
    parser.add_argument("--chunk", type=int, default=5, help="Dura\u00e7\u00e3o de cada captura em segundos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    translate_microphone(args.model, args.chunk)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
