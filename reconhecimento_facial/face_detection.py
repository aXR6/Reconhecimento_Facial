import argparse
import logging
import os
import tempfile
import threading
from typing import Optional, Tuple, List, Dict, Iterable

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

import cv2

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hf_hub_download = None

try:
    import mediapipe as mp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mp = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    YOLO = None

from .recognition import _social_search_background

logger = logging.getLogger(__name__)


def detect_faces(
    image_path: str,
    output_path: str = "output.jpg",
    use_hf: bool = False,
    hf_model: str = "mediapipe",
    *,
    scale: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
    show: bool = False,
    blur: bool = False,
    as_json: bool = False,
    save_db: bool = False,
    recognized: Optional[List[str]] = None,
    social_search: bool = False,
    sites: Iterable[str] | None = None,
    db_path: str | None = None,
) -> int | Dict[str, List[int]]:
    """Detecta rostos em ``image_path`` e salva resultado em ``output_path``.

    Se ``use_hf`` for ``True``, utiliza modelos da Hugging Face localmente
    (``mediapipe`` ou ``yolov8``) para complementar a detecção.

    Retorna o número total de rostos ou um dicionário com boxes quando
    ``as_json`` for ``True``. Pode desfocar as faces, exibir a imagem e
    salvar o resultado em um banco PostgreSQL. Quando ``social_search``
    é ``True``, cada rosto recortado é buscado nas redes sociais
    configuradas.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_neighbors, minSize=min_size
    )
    total_faces = len(faces)

    boxes = []
    for x, y, w, h in faces:
        if blur:
            roi = img[y : y + h, x : x + w]
            roi = cv2.GaussianBlur(roi, (99, 99), 30)
            img[y : y + h, x : x + w] = roi
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        boxes.append([int(x), int(y), int(w), int(h)])

    if use_hf:
        if hf_model == "yolov8":
            repo = os.getenv("YOLOV8_REPO", "jaredthejelly/yolov8s-face-detection")
            try:
                if hf_hub_download is None or YOLO is None:
                    raise RuntimeError("Dependências do YOLOv8 não instaladas")
                weight = hf_hub_download(repo, "YOLOv8-face-detection.pt")
                yolo = YOLO(weight)
                results = yolo(img, verbose=False)[0]
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    if blur:
                        roi = img[y1:y2, x1:x2]
                        roi = cv2.GaussianBlur(roi, (99, 99), 30)
                        img[y1:y2, x1:x2] = roi
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                total_faces += len(results.boxes)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao usar modelo YOLOv8: %s", exc)
        else:
            repo = os.getenv("MEDIAPIPE_REPO", "qualcomm/MediaPipe-Face-Detection")
            try:
                if mp is None:
                    raise RuntimeError("Dependências do MediaPipe não instaladas")
                mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)
                if res.detections:
                    for det in res.detections:
                        box = det.location_data.relative_bounding_box
                        x = int(box.xmin * img.shape[1])
                        y = int(box.ymin * img.shape[0])
                        w = int(box.width * img.shape[1])
                        h = int(box.height * img.shape[0])
                        if blur:
                            roi = img[y : y + h, x : x + w]
                            roi = cv2.GaussianBlur(roi, (99, 99), 30)
                            img[y : y + h, x : x + w] = roi
                        else:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        boxes.append([int(x), int(y), int(w), int(h)])
                    total_faces += len(res.detections)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao usar modelo MediaPipe: %s", exc)

    if recognized is None:
        try:
            from reconhecimento_facial.recognition import recognize_faces

            recognized = recognize_faces(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao reconhecer rostos: %s", exc)
            recognized = []

    if show:
        cv2.imshow("faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite(output_path, img)

    result = {"boxes": boxes, "count": total_faces}
    if social_search and boxes:
        _sites = list(sites) if sites else ["facebook"]
        for idx, (x, y, w, h) in enumerate(boxes):
            crop = img[y : y + h, x : x + w]
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, crop)
            name = ""
            if recognized and idx < len(recognized):
                name = recognized[idx]
            thr = threading.Thread(
                target=_social_search_background,
                args=(tmp.name, name, _sites, db_path),
                daemon=True,
            )
            thr.start()
    if save_db:
        try:
            from reconhecimento_facial.db import save_detection

            save_detection(
                image_path,
                total_faces,
                recognized=";".join(recognized or []),
                result_json=result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao salvar no banco: %s", exc)

    if as_json:
        return result
    return total_faces


def detect_faces_video(
    source: int | str = 0,
    output_path: str = "out.mp4",
    use_hf: bool = False,
    hf_model: str = "mediapipe",
    show: bool = False,
    blur: bool = False,
    show_info: bool = False,
    *,
    social_search: bool = False,
    sites: Iterable[str] | None = None,
    db_path: str | None = None,
) -> None:
    """Processa um vídeo ou webcam detectando rostos.

    Se ``social_search`` for ``True``, cada face encontrada é procurada nas
    redes sociais configuradas.
    """
    cap = cv2.VideoCapture(source)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tmp = "_tmp_frame.jpg"
        cv2.imwrite(tmp, frame)
        res = detect_faces(
            tmp,
            tmp,
            use_hf=use_hf,
            hf_model=hf_model,
            show=False,
            blur=blur,
            as_json=True,
            social_search=social_search,
            sites=sites,
            db_path=db_path,
        )
        processed = cv2.imread(tmp)
        if show_info:
            try:
                from .demographics_detection import detect_demographics

                for x, y, w, h in res.get("boxes", []):
                    crop = processed[y : y + h, x : x + w]
                    label = ""
                    info = detect_demographics(crop)
                    parts = []
                    gender = info.get("gender")
                    age = info.get("age")
                    ethnicity = info.get("ethnicity")
                    skin = info.get("skin")
                    if gender:
                        parts.append(gender)
                    if age:
                        parts.append(age)
                    if ethnicity:
                        parts.append(ethnicity)
                    if skin:
                        parts.append(skin)
                    label = ", ".join(parts)
                    if label:
                        cv2.putText(
                            processed,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao gerar info da webcam: %s", exc)
        if writer:
            writer.write(processed)
        if show:
            cv2.imshow("video", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta rostos em uma imagem.")
    parser.add_argument("--image", help="Caminho da imagem de entrada")
    parser.add_argument("--video", help="Arquivo de vídeo a processar")
    parser.add_argument("--camera", action="store_true", help="Usa webcam")
    parser.add_argument(
        "--output", default="output.jpg", help="Arquivo de saída com detecções"
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Utiliza modelo da Hugging Face (ver --model)",
    )
    parser.add_argument(
        "--model",
        choices=["mediapipe", "yolov8"],
        default="mediapipe",
        help="Modelo a ser usado com --hf",
    )
    parser.add_argument("--show", action="store_true", help="Exibe imagem")
    parser.add_argument("--blur", action="store_true", help="Desfoca faces")
    parser.add_argument(
        "--info", action="store_true", help="Exibe informacoes sobre as faces"
    )
    parser.add_argument("--json", action="store_true", help="Retorna JSON")
    parser.add_argument(
        "--save-db", action="store_true", help="Salva resultado no banco"
    )
    parser.add_argument(
        "--social-search", action="store_true", help="Busca rostos nas redes sociais"
    )
    parser.add_argument(
        "--site", action="append", default=["facebook"], help="Rede social para buscar"
    )
    parser.add_argument("--db", help="Diretório com imagens para busca social")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        if args.video or args.camera:
            src = 0 if args.camera else args.video
            detect_faces_video(
                src,
                args.output,
                use_hf=args.hf,
                hf_model=args.model,
                show=args.show,
                blur=args.blur,
                show_info=args.info,
                social_search=args.social_search,
                sites=args.site,
                db_path=args.db,
            )
            qtd = None
        else:
            qtd = detect_faces(
                args.image,
                args.output,
                use_hf=args.hf,
                hf_model=args.model,
                show=args.show,
                blur=args.blur,
                as_json=args.json,
                save_db=args.save_db,
                social_search=args.social_search,
                sites=args.site,
                db_path=args.db,
            )
    except FileNotFoundError as exc:
        logger.error(exc)
        return

    if qtd is None:
        logger.info("Processamento de vídeo finalizado")
    elif isinstance(qtd, dict):
        print(qtd)
    else:
        logger.info("Detectado(s) %s rosto(s). Resultado salvo em %s", qtd, args.output)


if __name__ == "__main__":
    main()
