import logging
import os
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
GENDER_LABELS = ["male", "female"]
RACE_LABELS = ["white", "black", "asian", "indian", "other"]


def _load_model() -> None:
    global _model, _device
    if _model is not None:
        return
    if hf_hub_download is None or MTCNN is None:
        logger.error("Required dependencies not installed")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        repo = os.getenv("FACEXFORMER_REPO", "kartiknarayan/facexformer")
        weight_path = hf_hub_download(repo, "ckpts/model.pt")
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to download facexformer weights: %s", exc)
        return
    model = FaceXFormer().to(device)
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict_backbone"])
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load facexformer model: %s", exc)
        return
    model.eval()
    _model = model
    _device = device


def detect_demographics(image_path: str) -> dict:
    """Detect age, gender and race using FaceXFormer."""
    _load_model()
    if _model is None:
        raise RuntimeError("FaceXFormer model not available")
    mtcnn = MTCNN(keep_all=False, device=_device)
    img = Image.open(image_path).convert("RGB")
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
    tensor = trans(face).unsqueeze(0).to(_device)

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

    (_, _, _, _, age_out, gender_out, race_out, _) = _model(tensor, labels, tasks)

    age_idx = int(torch.argmax(age_out, dim=1).item())
    gender_idx = int(torch.argmax(gender_out, dim=1).item())
    race_idx = int(torch.argmax(race_out, dim=1).item())

    return {
        "age": AGE_LABELS[age_idx] if age_idx < len(AGE_LABELS) else str(age_idx),
        "gender": GENDER_LABELS[gender_idx] if gender_idx < len(GENDER_LABELS) else str(gender_idx),
        "ethnicity": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
        "skin": RACE_LABELS[race_idx] if race_idx < len(RACE_LABELS) else str(race_idx),
    }
