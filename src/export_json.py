from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .utils import ensure_directory

LOGGER = logging.getLogger(__name__)


def export_report(patient_id: str, prediction: np.ndarray, image: np.ndarray, architecture: str, output_dir: str, confidence: float = 0.0) -> str:
    """Write a JSON report with tumour geometry summary and metadata."""
    mask = prediction.astype(bool)
    if mask.any():
        coords = np.argwhere(mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        largest_slice = int(np.argmax([np.sum(mask[:, :, z]) for z in range(mask.shape[2])]))
        volume_voxels = int(mask.sum())
        volume_mm3 = float(volume_voxels * 1.0)
        center = [float(np.mean(coords[:, 1])), float(np.mean(coords[:, 2])), float(np.mean(coords[:, 0]))]
    else:
        zmin = ymin = xmin = zmax = ymax = xmax = largest_slice = 0
        volume_voxels = 0
        volume_mm3 = 0.0
        center = [0.0, 0.0, 0.0]

    payload = {
        "patient_id": patient_id,
        "tumour_detected": bool(mask.any()),
        "tumour_volume_voxels": volume_voxels,
        "tumour_volume_mm3": volume_mm3,
        "tumour_center": center,
        "bounding_box": {
            "xmin": int(xmin),
            "xmax": int(xmax),
            "ymin": int(ymin),
            "ymax": int(ymax),
            "zmin": int(zmin),
            "zmax": int(zmax),
        },
        "largest_slice": largest_slice,
        "confidence": float(confidence),
        "model_architecture": architecture,
    }
    output_path = Path(output_dir) / f"{patient_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(output_path)
