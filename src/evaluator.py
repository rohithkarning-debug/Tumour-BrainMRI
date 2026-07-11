from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

from .predictor import SegmentationPredictor
from .utils import (
    compute_dice,
    compute_f1,
    compute_iou,
    compute_pixel_accuracy,
    compute_precision,
    compute_recall,
    ensure_directory,
    save_json,
)

LOGGER = logging.getLogger(__name__)


class SegmentationEvaluator:
    """Evaluate a segmentation model on a held-out test dataset."""

    def __init__(
        self,
        checkpoint_path: str,
        architecture: Optional[str] = None,
        results_dir: str = "results",
    ) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.predictor = SegmentationPredictor(checkpoint_path, architecture=architecture)
        self.model = self.predictor.model.to(self.device)
        self.model.eval()
        self.dice_loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        ce_weight = torch.tensor([1.0, 50.0], device=self.device)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weight)
        self.results_dir = Path(results_dir)
        ensure_directory(self.results_dir)
        ensure_directory(self.results_dir / "plots")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        if test_loader is None or len(test_loader) == 0:
            raise ValueError("No test samples available for evaluation.")

        sample_count = len(test_loader.dataset)
        total_loss = 0.0
        total_accuracy = 0.0
        total_inference_time = 0.0
        metrics_accumulator = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                if masks.ndim == 4:
                    masks = masks.unsqueeze(1)

                target = masks[:, 0].long().unsqueeze(1)
                start_time = time.perf_counter()
                logits = self.model(images)
                inference_time = time.perf_counter() - start_time
                total_inference_time += inference_time

                dice_loss = self.dice_loss_fn(logits, target)
                ce_loss = self.ce_loss_fn(logits, target.squeeze(1))
                loss = 0.5 * dice_loss + 0.5 * ce_loss
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                target_labels = target.squeeze(1)
                accuracy = (predictions == target_labels).float().mean().item()
                total_accuracy += accuracy

                preds_binary = (predictions > 0).float()
                targets_binary = (target_labels > 0).float()

                metrics_accumulator["dice"].append(compute_dice(preds_binary, targets_binary))
                metrics_accumulator["iou"].append(compute_iou(preds_binary, targets_binary))
                metrics_accumulator["precision"].append(compute_precision(preds_binary, targets_binary))
                metrics_accumulator["recall"].append(compute_recall(preds_binary, targets_binary))
                metrics_accumulator["f1"].append(compute_f1(preds_binary, targets_binary))

        result = {
            "test_loss": total_loss / max(1, len(test_loader)),
            "test_accuracy": total_accuracy / max(1, len(test_loader)),
            "dice_score": float(sum(metrics_accumulator["dice"]) / max(1, len(metrics_accumulator["dice"]))),
            "iou_score": float(sum(metrics_accumulator["iou"]) / max(1, len(metrics_accumulator["iou"]))),
            "precision": float(sum(metrics_accumulator["precision"]) / max(1, len(metrics_accumulator["precision"]))),
            "recall": float(sum(metrics_accumulator["recall"]) / max(1, len(metrics_accumulator["recall"]))),
            "f1_score": float(sum(metrics_accumulator["f1"]) / max(1, len(metrics_accumulator["f1"]))),
            "num_test_scans": sample_count,
            "average_inference_time_sec": total_inference_time / max(1, sample_count),
        }
        return result

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        save_json(self.results_dir / "metrics.json", metrics)

    def print_report(self, metrics: Dict[str, Any], training_history: Optional[Dict[str, Any]] = None) -> None:
        def last_value(key: str, default: float = 0.0) -> float:
            if not training_history or key not in training_history:
                return default
            value = training_history[key]
            return float(value[-1]) if isinstance(value, list) and value else float(value)

        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        if training_history is not None:
            print(f"Training Accuracy : {last_value('train_accuracy') * 100:.2f} %")
            print(f"Validation Accuracy : {last_value('val_accuracy') * 100:.2f} %")
            print(f"Test Accuracy : {metrics['test_accuracy'] * 100:.2f} %")
            print()
            print(f"Training Loss : {last_value('train_loss'):.4f}")
            print(f"Validation Loss : {last_value('val_loss'):.4f}")
            print(f"Test Loss : {metrics['test_loss']:.4f}")
        else:
            print(f"Test Accuracy : {metrics['test_accuracy'] * 100:.2f} %")
            print(f"Test Loss : {metrics['test_loss']:.4f}")
        print()
        print(f"Dice Score : {metrics['dice_score']:.4f}")
        print(f"IoU : {metrics['iou_score']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")
        print()
        print(f"Average Inference Time : {metrics['average_inference_time_sec']:.4f} sec")
        print(f"Number of Test Cases : {metrics['num_test_scans']}")
        print("=" * 50)
