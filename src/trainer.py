from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
from monai.losses import DiceCELoss
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from .model import get_model
from .utils import (
    compute_dice,
    compute_f1,
    compute_iou,
    compute_pixel_accuracy,
    compute_precision,
    compute_recall,
    ensure_directory,
    get_device,
    save_json,
)

LOGGER = logging.getLogger(__name__)


class SegmentationTrainer:
    """Train a 3D MONAI segmentation model for tumour segmentation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = get_device()
        self.model = get_model(
            architecture=self.config.get("architecture", "segresnet"),
            in_channels=self.config.get("in_channels", 1),
            out_channels=self.config.get("out_channels", 2),
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.get("learning_rate", 1e-4), weight_decay=1e-5)
        class_weights = self.config.get("class_weights", [0.25, 0.75])
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.7,
            lambda_ce=0.3,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=self.config.get("log_dir", "output/runs"))
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        self.checkpoint_path = Path(self.config.get("checkpoint_path", "models/best_model.pth")).expanduser().resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    def _load_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.best_dice = state.get("best_dice", 0.0)
            LOGGER.info("Loaded checkpoint from %s", checkpoint_path)

    def train(self, train_loader: Any, val_loader: Optional[Any]) -> Dict[str, Any]:
        checkpoint_path = self.config.get("resume_from")
        self._load_checkpoint(checkpoint_path)

        for epoch in range(self.config.get("epochs", 5)):
            train_loss, train_accuracy = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", train_accuracy, epoch)

            if val_loader is not None and len(val_loader) > 0:
                val_stats = self._validate(val_loader, epoch)
            else:
                val_stats = {
                    "val_loss": 0.0,
                    "val_accuracy": 0.0,
                    "val_dice": 0.0,
                    "val_iou": 0.0,
                    "val_precision": 0.0,
                    "val_recall": 0.0,
                    "val_f1": 0.0,
                }

            self.history["val_loss"].append(val_stats["val_loss"])
            self.history["val_accuracy"].append(val_stats["val_accuracy"])
            self.history["val_dice"].append(val_stats["val_dice"])
            self.history["val_iou"].append(val_stats["val_iou"])
            self.history["val_precision"].append(val_stats["val_precision"])
            self.history["val_recall"].append(val_stats["val_recall"])
            self.history["val_f1"].append(val_stats["val_f1"])

            self._save_checkpoint(epoch, val_stats["val_dice"])
            if self._should_stop(val_stats["val_dice"]):
                LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
                break

        self._save_training_plots()
        return self.history

    def _train_epoch(self, train_loader: Any) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        step_count = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if masks.ndim == 4:
                masks = masks.unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                target = masks[:, 0].long().unsqueeze(1)
                loss = self.loss_fn(outputs, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            predictions = torch.argmax(outputs, dim=1)
            target_labels = target.squeeze(1)
            accuracy = compute_pixel_accuracy(predictions, target_labels)
            total_loss += loss.item()
            total_accuracy += accuracy
            step_count += 1

        avg_loss = total_loss / max(1, step_count)
        avg_accuracy = total_accuracy / max(1, step_count)
        LOGGER.info("Epoch %s training loss: %.4f, accuracy: %.4f", step_count, avg_loss, avg_accuracy)
        return avg_loss, avg_accuracy

    def _validate(self, val_loader: Any, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        step_count = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                if masks.ndim == 4:
                    masks = masks.unsqueeze(1)

                outputs = self.model(images)
                target = masks[:, 0].long().unsqueeze(1)
                loss = self.loss_fn(outputs, target)
                predictions = torch.argmax(outputs, dim=1)
                target_labels = target.squeeze(1)
                preds_binary = (predictions > 0).float()
                targets_binary = (target_labels > 0).float()

                total_loss += loss.item()
                total_accuracy += compute_pixel_accuracy(predictions, target_labels)
                total_dice += compute_dice(preds_binary, targets_binary)
                total_iou += compute_iou(preds_binary, targets_binary)
                total_precision += compute_precision(preds_binary, targets_binary)
                total_recall += compute_recall(preds_binary, targets_binary)
                total_f1 += compute_f1(preds_binary, targets_binary)
                step_count += 1

        if step_count == 0:
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "val_dice": 0.0,
                "val_iou": 0.0,
                "val_precision": 0.0,
                "val_recall": 0.0,
                "val_f1": 0.0,
            }

        val_stats = {
            "val_loss": total_loss / step_count,
            "val_accuracy": total_accuracy / step_count,
            "val_dice": total_dice / step_count,
            "val_iou": total_iou / step_count,
            "val_precision": total_precision / step_count,
            "val_recall": total_recall / step_count,
            "val_f1": total_f1 / step_count,
        }

        self.writer.add_scalar("val/loss", val_stats["val_loss"], epoch)
        self.writer.add_scalar("val/accuracy", val_stats["val_accuracy"], epoch)
        self.writer.add_scalar("val/dice", val_stats["val_dice"], epoch)
        self.writer.add_scalar("val/iou", val_stats["val_iou"], epoch)
        self.writer.add_scalar("val/precision", val_stats["val_precision"], epoch)
        self.writer.add_scalar("val/recall", val_stats["val_recall"], epoch)
        self.writer.add_scalar("val/f1", val_stats["val_f1"], epoch)

        LOGGER.info(
            "Epoch %s validation loss: %.4f, accuracy: %.4f, dice: %.4f",
            epoch + 1,
            val_stats["val_loss"],
            val_stats["val_accuracy"],
            val_stats["val_dice"],
        )
        return val_stats

    def _save_checkpoint(self, epoch: int, val_dice: float) -> None:
        improved = val_dice > self.best_dice
        if improved:
            self.best_dice = val_dice
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "architecture": self.config.get("architecture", "segresnet"),
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(checkpoint, self.checkpoint_path)
        backup_path = self.checkpoint_path.with_suffix(".bak")
        torch.save(checkpoint, backup_path)
        if improved:
            LOGGER.info("Saved improved checkpoint to %s", self.checkpoint_path)
        else:
            LOGGER.info("Saved checkpoint to %s", self.checkpoint_path)

    def _should_stop(self, val_dice: float) -> bool:
        patience = self.config.get("early_stopping_patience", 3)
        return self.epochs_without_improvement >= patience

    def _save_training_plots(self) -> None:
        plot_dir = ensure_directory("results/plots")

        epochs = list(range(1, max(len(self.history["train_loss"]), len(self.history["val_loss"])) + 1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_loss"], marker="o", label="Train Loss")
        ax.plot(epochs, self.history["val_loss"], marker="o", label="Validation Loss")
        ax.set_title("Loss vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(epochs)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "loss_vs_epoch.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_accuracy"], marker="o", label="Train Accuracy")
        ax.plot(epochs, self.history["val_accuracy"], marker="o", label="Validation Accuracy")
        ax.set_title("Accuracy vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(epochs)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "accuracy_vs_epoch.png", dpi=150)
        plt.close(fig)

    def save_history(self, output_path: str) -> None:
        save_json(output_path, self.history)
