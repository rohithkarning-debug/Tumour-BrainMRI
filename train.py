from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.data_loader import CombinedDataModule, DataModule
from src.preprocess import Preprocessor, TrainingPreprocessor
from src.trainer import SegmentationTrainer
from src.utils import configure_logging, ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MONAI-based 3D brain tumour / healthy-brain MRI model"
    )
    parser.add_argument("--architecture", default="segresnet",
                        choices=["segresnet", "unet", "dynunet"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (keep 1 for 3-D volumes; use gradient accumulation instead)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Accumulate gradients over N batches before updating weights. "
                             "Effective batch = batch_size × steps. (default: 4)")
    parser.add_argument("--data-dir",
                        default="input/BraTS-MEN-RT-Train-v2",
                        help="Tumour (BraTS) dataset directory")
    parser.add_argument("--healthy-dir",
                        default="input/IXI_T1",
                        help="Healthy brain (IXI) dataset directory. "
                             "When provided the model trains on BOTH datasets.")
    parser.add_argument("--resume-from", default=None,
                        help="Optional checkpoint path to resume training from")
    parser.add_argument("--spatial-size", type=int, nargs=3,
                        default=(96, 96, 96),
                        help="Spatial size (D H W) to resize every volume to")
    parser.add_argument("--validation-split", type=float, default=0.10,
                        help="Fraction of patients used for validation")
    parser.add_argument("--test-split", type=float, default=0.10,
                        help="Fraction of patients used for test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible dataset splitting")
    parser.add_argument("--max-brats-patients", type=int, default=None,
                        help="Cap on BraTS patients used (useful for quick runs)")
    parser.add_argument("--max-ixi-patients", type=int, default=None,
                        help="Cap on IXI healthy patients used")
    parser.add_argument("--init-filters", type=int, default=32,
                        help="SegResNet initial filter count. 32=CPU-friendly (fast), 64=GPU-quality (4x slower)")
    parser.add_argument("--ce-weight", type=float, default=100.0,
                        help="(Legacy) CrossEntropy tumour weight — now replaced by Tversky β")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warm-up epochs before cosine annealing")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Peak learning rate after warm-up")
    parser.add_argument("--lr-restart-period", type=int, default=20,
                        help="T_0 for CosineAnnealingWarmRestarts (epochs per restart)")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Epochs without val-Dice improvement before stopping")
    # Legacy alias kept for backwards compatibility
    parser.add_argument("--max-patients", type=int, default=None,
                        help="(Legacy) Alias for --max-brats-patients")
    return parser.parse_args()


def clean_generated_outputs() -> None:
    logger = logging.getLogger(__name__)
    generated_dirs = [
        PROJECT_ROOT / "output",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "runs",
        PROJECT_ROOT / "overlays",
        PROJECT_ROOT / "json",
    ]
    for path_obj in generated_dirs:
        if path_obj.exists():
            try:
                shutil.rmtree(path_obj)
            except Exception as e:
                logger.warning("Could not remove directory %s: %s", path_obj, e)
    for model_file in ["models/best_model.pth", "best_model.pth"]:
        p = PROJECT_ROOT / model_file
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                logger.warning("Could not remove file %s: %s", p, e)


def main() -> None:
    args = parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    # --max-patients is a legacy alias for --max-brats-patients
    max_brats = args.max_brats_patients or args.max_patients

    logger.info(
        "Starting training | arch=%s | epochs=%d | spatial_size=%s",
        args.architecture, args.epochs, args.spatial_size,
    )

    if not args.resume_from:
        clean_generated_outputs()

    ensure_directory(PROJECT_ROOT / "output")
    ensure_directory(PROJECT_ROOT / "models")
    ensure_directory(PROJECT_ROOT / "results" / "plots")

    # ── Resolve paths ──────────────────────────────────────────────────────
    brats_dir = Path(args.data_dir)
    if not brats_dir.is_absolute():
        brats_dir = PROJECT_ROOT / brats_dir

    healthy_dir = Path(args.healthy_dir) if args.healthy_dir else None
    if healthy_dir and not healthy_dir.is_absolute():
        healthy_dir = PROJECT_ROOT / healthy_dir

    spatial_size = tuple(args.spatial_size)

    # ── Preprocessors (augmented for train, deterministic for eval) ────────
    train_preprocessor = TrainingPreprocessor(spatial_size=spatial_size)
    eval_preprocessor = Preprocessor(spatial_size=spatial_size)

    # ── DataModule ─────────────────────────────────────────────────────────
    if healthy_dir and healthy_dir.exists():
        logger.info(
            "Using CombinedDataModule: BraTS=%s | IXI=%s", brats_dir, healthy_dir
        )
        data_module = CombinedDataModule(
            brats_dir=str(brats_dir),
            ixi_dir=str(healthy_dir),
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            test_split=args.test_split,
            train_transform=train_preprocessor,
            eval_transform=eval_preprocessor,
            max_brats_patients=max_brats,
            max_ixi_patients=args.max_ixi_patients,
            random_seed=args.seed,
        )
    else:
        logger.info("No healthy-dir found; training on BraTS only: %s", brats_dir)
        data_module = DataModule(
            str(brats_dir),
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            test_split=args.test_split,
            train_transform=train_preprocessor,
            eval_transform=eval_preprocessor,
            max_patients=max_brats,
            random_seed=args.seed,
        )

    # ── Trainer config ─────────────────────────────────────────────────────
    config = {
        "architecture":               args.architecture,
        "epochs":                     args.epochs,
        "batch_size":                 args.batch_size,
        "learning_rate":              args.learning_rate,
        "warmup_epochs":              args.warmup_epochs,
        "lr_restart_period":          args.lr_restart_period,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "init_filters":               args.init_filters,
        "checkpoint_path":            str(PROJECT_ROOT / "models" / "best_model.pth"),
        "log_dir":                    str(PROJECT_ROOT / "output" / "runs"),
        "resume_from":                args.resume_from,
        "early_stopping_patience":    args.early_stopping_patience,
    }

    trainer = SegmentationTrainer(config)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader), len(val_loader) if val_loader else 0,
    )

    trainer.train(train_loader, val_loader)
    trainer.save_history(str(PROJECT_ROOT / "results" / "training_history.json"))

    logger.info("Training complete. Best checkpoint → %s", config["checkpoint_path"])
    logger.info("Training history → %s", PROJECT_ROOT / "results" / "training_history.json")


if __name__ == "__main__":
    main()
