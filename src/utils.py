import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure a simple root logger for the project."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array on CPU."""
    return tensor.detach().cpu().numpy()


def to_one_hot(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Convert a binary mask to one-hot representation."""
    mask = mask.astype(np.int64)
    one_hot = np.zeros((num_classes, *mask.shape), dtype=np.float32)
    one_hot[1] = (mask > 0).astype(np.float32)
    one_hot[0] = (mask == 0).astype(np.float32)
    return one_hot


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a JSON file."""
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    """Save a JSON file."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def compute_dice(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Dice coefficient for binary segmentation targets."""
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    return 2.0 * intersection / union if union > 0 else 0.0


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Intersection over Union for binary segmentation."""
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum().item()
    union = (preds + targets - preds * targets).sum().item()
    return intersection / union if union > 0 else 0.0


def compute_precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute precision for binary segmentation."""
    preds = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute recall for binary segmentation."""
    preds = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute F1 score for binary segmentation."""
    precision = compute_precision(preds, targets)
    recall = compute_recall(preds, targets)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute pixel accuracy for binary segmentation."""
    return (preds == targets).float().mean().item()


def list_patient_ids(input_dir: str | os.PathLike[str]) -> List[str]:
    """List patient IDs from the BraTS input directory."""
    root = Path(input_dir)
    patient_ids = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            patient_ids.append(path.name)
    return patient_ids
