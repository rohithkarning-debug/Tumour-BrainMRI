from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
from monai.losses import DiceFocalLoss, TverskyLoss
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

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


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTrainer:
    """Train a 3-D MONAI segmentation model for brain tumour detection.

    Loss strategy
    ─────────────
    We combine two complementary losses:

    1. DiceFocalLoss  (λ_dice=0.5, λ_focal=0.5, γ=2.0)
       • Dice term  → maximises volumetric overlap (global metric)
       • Focal term → down-weights easy background voxels so the model
                      focuses training signal on hard / rare tumour voxels.
                      γ=2 is the standard value from the original Focal paper.

    2. TverskyLoss  (α=0.3, β=0.7)
       • Penalises *false negatives* 2.3× more than false positives.
       • This pushes recall up — critical because missing a tumour is
         far worse clinically than a false alarm.

    Final loss = 0.5 · DiceFocal + 0.5 · Tversky

    This combination is used by top-ranking BraTS challenge entries and
    consistently outperforms plain Dice + CrossEntropy by 3-8 Dice points.

    Optimiser & schedule
    ────────────────────
    AdamW with weight decay (1e-5) + linear warm-up for the first
    `warmup_epochs` epochs, then CosineAnnealingWarmRestarts.
    Warm restarts escape local minima that plain cosine decay misses.

    Gradient accumulation
    ─────────────────────
    With batch_size=1 (memory constraint), we accumulate gradients over
    `gradient_accumulation_steps` batches before updating weights.
    Effective batch size = batch_size × accumulation_steps.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = get_device()
        self.accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 4)))

        self.model = get_model(
            architecture=config.get('architecture', 'segresnet'),
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 2),
            init_filters=config.get('init_filters', 64),
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 2e-4),
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ── Loss 1: DiceFocalLoss ─────────────────────────────────────────────
        self.dice_focal_loss_fn = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            gamma=2.0,          # focal exponent — standard value
            lambda_dice=0.5,
            lambda_focal=0.5,
        )

        # ── Loss 2: TverskyLoss ───────────────────────────────────────────────
        # alpha = FP weight, beta = FN weight
        # beta > alpha → penalise missed tumour more than false alarm
        self.tversky_loss_fn = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=0.3,          # FP penalty (low)
            beta=0.7,           # FN penalty (high → drives recall up)
        )

        # ── Learning-rate schedule ────────────────────────────────────────────
        total_epochs   = config.get("epochs", 100)
        warmup_epochs  = config.get("warmup_epochs", 5)
        restart_period = config.get("lr_restart_period", 20)   # T_0

        # Phase 1: linear warm-up
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            return 1.0   # hand off to cosine scheduler after warmup

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # Phase 2: cosine with warm restarts (fires AFTER warmup)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=restart_period,    # restart every T_0 epochs
            T_mult=2,              # double the period after each restart
            eta_min=1e-7,
        )

        self._warmup_done = False
        self._warmup_epochs = warmup_epochs

        self.scaler = GradScaler(device=self.device.type)

        log_dir = config.get("log_dir", "output/runs")
        if _HAS_TB:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        self.best_dice               = 0.0
        self.epochs_without_improvement = 0
        self._seen_nonzero_dice      = False

        self.checkpoint_path = (
            Path(config.get("checkpoint_path", "models/best_model.pth"))
            .expanduser()
            .resolve()
        )
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss":       [],
            "val_loss":         [],
            "train_accuracy":   [],
            "val_accuracy":     [],
            "val_dice":         [],
            "val_iou":          [],
            "val_precision":    [],
            "val_recall":       [],
            "val_f1":           [],
            "val_cls_accuracy": [],
            "lr":               [],
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, train_loader: Any, val_loader: Optional[Any]) -> Dict[str, Any]:
        resume_path = self.config.get("resume_from")
        self._load_checkpoint(resume_path)

        for epoch in range(self.config.get("epochs", 100)):
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            if self.writer:
                self.writer.add_scalar("train/loss",     train_loss,  epoch)
                self.writer.add_scalar("train/accuracy", train_acc,   epoch)
                self.writer.add_scalar("train/lr",       current_lr,  epoch)

            if val_loader and len(val_loader) > 0:
                val_stats = self._validate(val_loader, epoch)
            else:
                val_stats = self._empty_val_stats()

            for k in ("val_loss", "val_accuracy", "val_dice", "val_iou",
                      "val_precision", "val_recall", "val_f1", "val_cls_accuracy"):
                self.history[k].append(val_stats[k])

            # ── LR step ──────────────────────────────────────────────────────
            if epoch < self._warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if not self._warmup_done:
                    self._warmup_done = True
                    LOGGER.info("Warm-up complete — switching to cosine annealing")
                self.cosine_scheduler.step(epoch - self._warmup_epochs)

            self._save_checkpoint(epoch, val_stats["val_dice"])
            if self._should_stop(val_stats["val_dice"]):
                LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        self._save_training_plots()
        return self.history

    def save_history(self, output_path: str) -> None:
        save_json(output_path, self.history)

    # ── Private: training loop ────────────────────────────────────────────────

    def _compute_loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined DiceFocal + Tversky loss."""
        dice_focal = self.dice_focal_loss_fn(outputs, target)
        tversky    = self.tversky_loss_fn(outputs, target)
        return 0.5 * dice_focal + 0.5 * tversky

    def _train_epoch(self, train_loader: Any, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss     = 0.0
        total_accuracy = 0.0
        n_batches      = len(train_loader)
        optimizer_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            is_last_batch = (batch_idx == n_batches - 1)

            images = batch["image"].to(self.device)
            masks  = batch["mask"].to(self.device)
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if masks.ndim == 4:
                masks = masks.unsqueeze(1)

            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                target  = masks[:, 0].long().unsqueeze(1)
                # Scale loss by accumulation steps so gradients are averaged
                loss    = self._compute_loss(outputs, target) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            # Log unscaled loss for readability
            total_loss += loss.item() * self.accumulation_steps

            with torch.no_grad():
                predictions  = torch.argmax(outputs, dim=1)
                target_labels = target.squeeze(1)
                total_accuracy += compute_pixel_accuracy(predictions, target_labels)

            # ── Weight update ─────────────────────────────────────────────────
            should_update = (
                (batch_idx + 1) % self.accumulation_steps == 0
                or is_last_batch
            )
            if should_update:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        avg_loss     = total_loss     / max(1, n_batches)
        avg_accuracy = total_accuracy / max(1, n_batches)
        current_lr   = self.optimizer.param_groups[0]["lr"]

        LOGGER.info(
            "Epoch %3d | loss=%.4f | acc=%.4f | lr=%.2e | opt_steps=%d",
            epoch + 1, avg_loss, avg_accuracy, current_lr, optimizer_steps,
        )
        return avg_loss, avg_accuracy

    # ── Private: validation loop ──────────────────────────────────────────────

    @staticmethod
    def _scan_level_cls_accuracy(
        predictions: torch.Tensor,
        has_tumour_batch: Any,
    ) -> float:
        """Scan-level tumour/healthy classification accuracy."""
        batch_size = predictions.shape[0]
        correct = 0
        for i in range(batch_size):
            pred_positive = (predictions[i] > 0).any().item()
            if torch.is_tensor(has_tumour_batch[i]):
                gt_positive = bool(has_tumour_batch[i].item())
            else:
                gt_positive = bool(has_tumour_batch[i])
            if pred_positive == gt_positive:
                correct += 1
        return correct / batch_size if batch_size > 0 else 0.0

    def _validate(self, val_loader: Any, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals = {k: 0.0 for k in (
            "loss", "accuracy", "dice", "iou",
            "precision", "recall", "f1", "cls_accuracy"
        )}
        step_count = 0

        with torch.no_grad():
            for batch in val_loader:
                images     = batch["image"].to(self.device)
                masks      = batch["mask"].to(self.device)
                has_tumour = batch.get("has_tumour", None)
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                if masks.ndim == 4:
                    masks = masks.unsqueeze(1)

                outputs      = self.model(images)
                target       = masks[:, 0].long().unsqueeze(1)
                loss         = self._compute_loss(outputs, target)
                predictions  = torch.argmax(outputs, dim=1)
                target_labels = target.squeeze(1)
                preds_bin    = (predictions > 0).float()
                tgts_bin     = (target_labels > 0).float()

                totals["loss"]      += loss.item()
                totals["accuracy"]  += compute_pixel_accuracy(predictions, target_labels)
                totals["dice"]      += compute_dice(preds_bin, tgts_bin)
                totals["iou"]       += compute_iou(preds_bin, tgts_bin)
                totals["precision"] += compute_precision(preds_bin, tgts_bin)
                totals["recall"]    += compute_recall(preds_bin, tgts_bin)
                totals["f1"]        += compute_f1(preds_bin, tgts_bin)

                if has_tumour is not None:
                    totals["cls_accuracy"] += self._scan_level_cls_accuracy(
                        predictions, has_tumour
                    )
                step_count += 1

        if step_count == 0:
            return self._empty_val_stats()

        val_stats = {
            f"val_{k}": totals[k] / step_count
            for k in totals
        }

        if self.writer:
            for k, v in val_stats.items():
                self.writer.add_scalar(k.replace("val_", "val/"), v, epoch)

        LOGGER.info(
            "Epoch %3d | val_loss=%.4f | val_acc=%.4f | val_dice=%.4f "
            "| val_iou=%.4f | val_prec=%.4f | val_rec=%.4f | val_f1=%.4f | val_cls_acc=%.4f",
            epoch + 1,
            val_stats["val_loss"],        val_stats["val_accuracy"],
            val_stats["val_dice"],        val_stats["val_iou"],
            val_stats["val_precision"],   val_stats["val_recall"],
            val_stats["val_f1"],          val_stats["val_cls_accuracy"],
        )
        return val_stats

    @staticmethod
    def _empty_val_stats() -> Dict[str, float]:
        return {
            "val_loss":         0.0,
            "val_accuracy":     0.0,
            "val_dice":         0.0,
            "val_iou":          0.0,
            "val_precision":    0.0,
            "val_recall":       0.0,
            "val_f1":           0.0,
            "val_cls_accuracy": 0.0,
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _load_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.best_dice = state.get("best_dice", 0.0)
            LOGGER.info("Resumed from checkpoint %s", checkpoint_path)

    def _save_checkpoint(self, epoch: int, val_dice: float) -> None:
        improved = val_dice > self.best_dice
        if improved:
            self.best_dice = val_dice
            self.epochs_without_improvement = 0
            self._seen_nonzero_dice = True
        else:
            if self._seen_nonzero_dice:
                self.epochs_without_improvement += 1
            else:
                self.epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice":            self.best_dice,
            "architecture":         self.config.get("architecture", "segresnet"),
            "epoch":                epoch,
            "history":              self.history,
        }
        if improved:
            torch.save(checkpoint, self.checkpoint_path)
            LOGGER.info(
                "✓ New best Dice=%.4f — saved to %s", val_dice, self.checkpoint_path
            )
            # ── Backup to Google Drive (if running in Colab) ──────────────────
            self._backup_to_drive(checkpoint)

        # Always keep a rolling backup locally
        torch.save(checkpoint, self.checkpoint_path.with_suffix(".bak"))

    def _backup_to_drive(self, checkpoint: dict) -> None:
        """Copy checkpoint + history to Google Drive when running in Colab."""
        import shutil
        drive_dir = "/content/drive/MyDrive/Tumour-MRI-Data"
        if not os.path.isdir("/content/drive/MyDrive"):
            return  # Not running in Colab / Drive not mounted — skip silently
        try:
            os.makedirs(drive_dir, exist_ok=True)
            # Save model checkpoint
            drive_ckpt = os.path.join(drive_dir, "best_model.pth")
            shutil.copy2(str(self.checkpoint_path), drive_ckpt)
            # Save training history alongside it
            save_json(
                os.path.join(drive_dir, "training_history.json"),
                self.history,
            )
            LOGGER.info("✓ Checkpoint + history backed up to Google Drive")
        except Exception as exc:
            LOGGER.warning("Could not back up to Google Drive: %s", exc)

    def _should_stop(self, val_dice: float) -> bool:
        patience = self.config.get("early_stopping_patience", 15)
        return self.epochs_without_improvement >= patience

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _save_training_plots(self) -> None:
        plot_dir = ensure_directory("results/plots")
        epochs   = list(range(1, len(self.history["train_loss"]) + 1))

        def _plot(keys_labels, title, ylabel, fname):
            fig, ax = plt.subplots(figsize=(10, 5))
            for key, label in keys_labels:
                data = self.history.get(key, [])
                if data:
                    ax.plot(epochs[:len(data)], data, marker="o", markersize=3, label=label)
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(); fig.tight_layout()
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)

        _plot([("train_loss", "Train"), ("val_loss", "Val")],
              "Loss vs Epoch", "Loss", "loss_vs_epoch.png")
        _plot([("train_accuracy", "Train"), ("val_accuracy", "Val")],
              "Pixel Accuracy vs Epoch", "Accuracy", "accuracy_vs_epoch.png")
        _plot([("val_dice", "Dice"), ("val_iou", "IoU"), ("val_f1", "F1"),
               ("val_cls_accuracy", "Cls Acc")],
              "Segmentation & Classification Metrics", "Score", "metrics_vs_epoch.png")
        _plot([("val_precision", "Precision"), ("val_recall", "Recall")],
              "Precision & Recall vs Epoch", "Score", "precision_recall_vs_epoch.png")
        _plot([("lr", "LR")],
              "Learning Rate Schedule", "LR", "lr_schedule.png")
