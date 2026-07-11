from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .model import get_model
from .preprocess import Preprocessor
from .utils import get_device

LOGGER = logging.getLogger(__name__)


class SegmentationPredictor:
    """Load a checkpoint and run inference for a single patient volume."""

    def __init__(
        self,
        checkpoint_path: str,
        architecture: Optional[str] = None,
        spatial_size: tuple[int, int, int] = (64, 64, 64),
        threshold: float = 0.75,
        min_component_voxels: int = 10,
    ) -> None:
        self.device = get_device()
        self.checkpoint_path = checkpoint_path
        if not Path(self.checkpoint_path).is_absolute():
            self.checkpoint_path = str(Path(__file__).resolve().parents[1] / self.checkpoint_path)
        self.architecture = architecture
        self.spatial_size = spatial_size
        self.threshold = threshold
        self.min_component_voxels = min_component_voxels
        self.preprocessor = Preprocessor(spatial_size=self.spatial_size)
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> torch.nn.Module:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        architecture = self.architecture or checkpoint.get("architecture", "segresnet")
        self.architecture = architecture
        model = get_model(architecture=architecture, in_channels=1, out_channels=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _prepare_input(self, image: np.ndarray, spatial_size: tuple[int, int, int] | None = None) -> torch.Tensor:
        target_shape = spatial_size or self.spatial_size
        sample = {
            "image": image.astype(np.float32),
            "mask": np.zeros_like(image, dtype=np.float32),
            "patient_id": "",
        }
        preprocessor = getattr(self, "preprocessor", None)
        if preprocessor is None or tuple(preprocessor.spatial_size) != target_shape:
            preprocessor = Preprocessor(spatial_size=target_shape)
        prepared = preprocessor(sample)["image"]
        return prepared.unsqueeze(0)

    def _postprocess_prediction(self, probabilities: torch.Tensor) -> np.ndarray:
        probability_map = probabilities.detach().cpu().numpy()
        binary = probability_map > self.threshold
        if not np.any(binary):
            return np.zeros_like(binary, dtype=np.uint8)

        binary = ndimage.binary_opening(binary, structure=np.ones((3, 3, 3))).astype(np.uint8)
        labels, num_labels = ndimage.label(binary)
        if num_labels == 0:
            return np.zeros_like(binary, dtype=np.uint8)

        component_sizes = np.bincount(labels.ravel())
        keep = np.zeros_like(binary, dtype=np.uint8)
        for label_id in range(1, num_labels + 1):
            if component_sizes[label_id] >= self.min_component_voxels:
                keep[labels == label_id] = 1
        return keep.astype(np.uint8)

    def predict(self, image: np.ndarray, patient_id: str) -> Dict[str, Any]:
        original_shape = tuple(image.shape)
        tensor = self._prepare_input(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[:, 1].squeeze(0)

        probabilities = torch.nn.functional.interpolate(
            probabilities.unsqueeze(0).unsqueeze(0).float(),
            size=original_shape,
            mode="trilinear",
        ).squeeze(0).squeeze(0)

        prediction = self._postprocess_prediction(probabilities)
        return {"prediction": prediction, "probabilities": probabilities.cpu().numpy(), "patient_id": patient_id}
