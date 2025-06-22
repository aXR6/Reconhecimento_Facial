import logging
import os
import inspect
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MTCNN = None

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hf_hub_download = None

from .models.facexformer import FaceXFormer

logger = logging.getLogger(__name__)

_model = None
_device = None

AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+",
]
GENDER_LABELS = ["masculino", "feminino"]
RACE_LABELS = ["branco", "negro", "asiático", "indiano", "outro"]

# 40 CelebA attribute labels used by FaceXFormer
ATTRIBUTE_LABELS = [
    "Sombra_de_Barba",
    "Sobrancelhas_Arqueadas",
    "Atraente",
    "Bolsas_Sob_os_Olhos",
    "Careca",
    "Franja",
    "Lábios_Grandes",
    "Nariz_Grande",
    "Cabelo_Preto",
    "Cabelo_Loiro",
    "Embaçado",
    "Cabelo_Castanho",
    "Sobrancelhas_Espessas",
    "Gordinho",
    "Queixo_Duplo",
    "Óculos",
    "Cavanhaque",
    "Cabelo_Grisalho",
    "Maquiagem_Pesada",
    "Maçãs_do_Rosto_Altas",
    "Masculino",
    "Boca_Entreaberta",
    "Bigode",
    "Olhos_Estreitos",
    "Sem_Barba",
    "Rosto_Oval",
    "Pele_Pálida",
    "Nariz_Pontudo",
    "Calvície_Frontal",
    "Bochechas_Rosadas",
    "Costeletas",
    "Sorrindo",
    "Cabelo_Liso",
    "Cabelo_Ondulado",
    "Usando_Brinco",
    "Usando_Chapéu",
    "Usando_Batom",
    "Usando_Colar",
    "Usando_Gravata",
    "Jovem",
]


def _prepare_face(image: Any) -> torch.Tensor:
    """Return a normalized face tensor detected in ``image``."""
    mtcnn = MTCNN(keep_all=False, device=_device)
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            img = Image.fromarray(image[:, :, ::-1])
        else:
            img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.open(image).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        raise RuntimeError("no face detected")
    x1, y1, x2, y2 = boxes[0]
    w, h = img.size
    dx = (x2 - x1) * 0.2
    dy = (y2 - y1) * 0.2
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    face = img.crop((int(x1), int(y1), int(x2), int(y2)))
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return trans(face).unsqueeze(0).to(_device)


def _load_model() -> None:
    global _model, _device
    if _model is not None:
        return
    if hf_hub_download is None or MTCNN is None:
        logger.error("Required dependencies not installed")
        return
    from ..device import torch_device, set_device

    device = torch_device()
    try:
        repo = os.getenv("FACEXFORMER_REPO", "kartiknarayan/facexformer")
        weight_path = hf_hub_download(repo, "ckpts/model.pt")
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to download facexformer weights: %s", exc)
        return

    model = FaceXFormer()
    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
        kwargs = {"assign": True} if "assign" in inspect.signature(model.load_state_dict).parameters else {}
        model.load_state_dict(checkpoint["state_dict_backbone"], **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load facexformer model: %s", exc)
        return

    try:
        model.to(device)
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            logger.error("GPU sem memoria. Usando CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_device("cpu")
            device = "cpu"
            model.to(device)
        else:
            raise
    model.eval()
    _model = model
    _device = device


def detect_demographics(image: Any) -> dict:
    """Detect age, gender and race using FaceXFormer.

    Parameters
    ----------
    image:
        Path to an image or an array (BGR) representing the image.
    """
    _load_model()
    if _model is None:
        raise RuntimeError("FaceXFormer model not available")
    tensor = _prepare_face(image)

    labels = {
        "segmentation": torch.zeros((1, 224, 224), device=_device),
        "lnm_seg": torch.zeros((1, 5, 2), device=_device),
        "landmark": torch.zeros((1, 68, 2), device=_device),
        "headpose": torch.zeros((1, 3), device=_device),
        "attribute": torch.zeros((1, 40), device=_device),
        "a_g_e": torch.zeros((1, 3), device=_device),
        "visibility": torch.zeros((1, 29), device=_device),
    }
    tasks = torch.tensor([4], device=_device)

    try:
        (_, _, _, _, age_out, gender_out, race_out, _) = _model(tensor, labels, tasks)
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            logger.error("GPU sem memoria. Migrando para CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_device("cpu")
            _model.to("cpu")
            global _device
            _device = "cpu"
            tensor = _prepare_face(image)
            labels = {
                "segmentation": torch.zeros((1, 224, 224), device=_device),
                "lnm_seg": torch.zeros((1, 5, 2), device=_device),
                "landmark": torch.zeros((1, 68, 2), device=_device),
                "headpose": torch.zeros((1, 3), device=_device),
                "attribute": torch.zeros((1, 40), device=_device),
                "a_g_e": torch.zeros((1, 3), device=_device),
                "visibility": torch.zeros((1, 29), device=_device),
            }
            tasks = torch.tensor([4], device=_device)
            (_, _, _, _, age_out, gender_out, race_out, _) = _model(
                tensor, labels, tasks
            )
        else:
            raise

    age_idx = int(torch.argmax(age_out, dim=1).item())
    gender_idx = int(torch.argmax(gender_out, dim=1).item())
    race_idx = int(torch.argmax(race_out, dim=1).item())

    return {
        "age": AGE_LABELS[age_idx] if age_idx < len(AGE_LABELS) else str(age_idx),
        "gender": GENDER_LABELS[gender_idx] if gender_idx < len(GENDER_LABELS) else str(gender_idx),
        "ethnicity": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
        "skin": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
    }


def analyze_face(image: Any) -> dict:
    """Return multiple facial analysis predictions using FaceXFormer."""
    _load_model()
    if _model is None:
        raise RuntimeError("FaceXFormer model not available")

    tensor = _prepare_face(image)

    labels = {
        "segmentation": torch.zeros((1, 224, 224), device=_device),
        "lnm_seg": torch.zeros((1, 5, 2), device=_device),
        "landmark": torch.zeros((1, 68, 2), device=_device),
        "headpose": torch.zeros((1, 3), device=_device),
        "attribute": torch.zeros((1, 40), device=_device),
        "a_g_e": torch.zeros((1, 3), device=_device),
        "visibility": torch.zeros((1, 29), device=_device),
    }

    def _run(tasks: torch.Tensor):
        nonlocal tensor, labels
        try:
            return _model(tensor, labels, tasks)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                logger.error("GPU sem memoria. Migrando para CPU")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                set_device("cpu")
                _model.to("cpu")
                global _device
                _device = "cpu"
                tensor = _prepare_face(image)
                labels = {
                    "segmentation": torch.zeros((1, 224, 224), device=_device),
                    "lnm_seg": torch.zeros((1, 5, 2), device=_device),
                    "landmark": torch.zeros((1, 68, 2), device=_device),
                    "headpose": torch.zeros((1, 3), device=_device),
                    "attribute": torch.zeros((1, 40), device=_device),
                    "a_g_e": torch.zeros((1, 3), device=_device),
                    "visibility": torch.zeros((1, 29), device=_device),
                }
                tasks = tasks.to(_device)
                return _model(tensor, labels, tasks)
            raise

    result: dict[str, Any] = {}

    # segmentation mask
    tasks = torch.tensor([0], device=_device)
    (_, _, _, _, _, _, _, seg) = _run(tasks)
    result["segmentation"] = seg.argmax(dim=1).squeeze(0).cpu().numpy().tolist()

    # landmarks
    tasks = torch.tensor([1], device=_device)
    (lm, _, _, _, _, _, _, _) = _run(tasks)
    result["landmarks"] = lm.view(68, 2).cpu().tolist()

    # headpose
    tasks = torch.tensor([2], device=_device)
    (_, hp, _, _, _, _, _, _) = _run(tasks)
    result["headpose"] = {
        "pitch": float(hp[0, 0].item()),
        "yaw": float(hp[0, 1].item()),
        "roll": float(hp[0, 2].item()),
    }

    # attributes
    tasks = torch.tensor([3], device=_device)
    (_, _, attr, _, _, _, _, _) = _run(tasks)
    scores = attr.squeeze(0)
    attrs = {}
    for idx, name in enumerate(ATTRIBUTE_LABELS):
        if idx < scores.numel():
            attrs[name] = bool(scores[idx] > 0)
    result["attributes"] = attrs

    # age/gender/race
    tasks = torch.tensor([4], device=_device)
    (_, _, _, _, age_out, gender_out, race_out, _) = _run(tasks)
    age_idx = int(torch.argmax(age_out, dim=1).item())
    gender_idx = int(torch.argmax(gender_out, dim=1).item())
    race_idx = int(torch.argmax(race_out, dim=1).item())
    result.update(
        {
            "age": AGE_LABELS[age_idx] if age_idx < len(AGE_LABELS) else str(age_idx),
            "gender": GENDER_LABELS[gender_idx] if gender_idx < len(GENDER_LABELS) else str(gender_idx),
            "ethnicity": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
            "skin": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
        }
    )

    # visibility
    tasks = torch.tensor([5], device=_device)
    (_, _, _, vis, _, _, _, _) = _run(tasks)
    result["visibility"] = int(vis.argmax(dim=1).item())

    return result
