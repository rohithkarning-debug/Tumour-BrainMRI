"""
inference.py  —  Brain Tumour Detection & Full Diagnostic Report
================================================================
Usage (from the project root):

  python inference.py --input path/to/brain.nii.gz

  # or a folder containing a *_t1c.nii.gz file (BraTS format):
  python inference.py --input input/BraTS-MEN-RT-Train-v2/BraTS-MEN-RT-0002-1/

Outputs
-------
  - A clear VERDICT printed to the terminal (TUMOUR DETECTED / NO TUMOUR)
  - All numerical values: confidence, tumour volume, Dice (if GT available),
    IoU, precision, recall, F1, voxel counts, etc.
  - A JSON report saved to output/json/<patient_id>.json
  - An overlay slice saved to output/overlays/<patient_id>.png
  - Optionally the raw prediction mask as output/predictions/<patient_id>.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.model import get_model
from src.preprocess import Preprocessor
from src.utils import configure_logging, ensure_directory, get_device

PROJECT_ROOT = Path(__file__).resolve().parent

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Brain Tumour Detection — pass ANY brain MRI, get a full diagnostic report"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a .nii / .nii.gz MRI file  OR  a patient folder "
             "(will auto-find the T1c file inside)",
    )
    p.add_argument(
        "--checkpoint", default="models/best_model.pth",
        help="Path to the trained model checkpoint  (default: models/best_model.pth)",
    )
    p.add_argument(
        "--architecture", default=None,
        help="Architecture override — auto-detected from checkpoint when omitted",
    )
    p.add_argument(
        "--spatial-size", type=int, nargs=3, default=(96, 96, 96),
        help="Spatial size used when the model was trained  (D H W, default: 96 96 96)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.50,
        help="Tumour-class probability threshold for binary mask  (default: 0.50)",
    )
    p.add_argument(
        "--min-component-voxels", type=int, default=10,
        help="Remove connected components smaller than this many voxels",
    )
    p.add_argument(
        "--patient-id", default=None,
        help="Label used in reports/filenames — auto-inferred when omitted",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory for JSON, overlay, and prediction outputs",
    )
    p.add_argument(
        "--save-prediction", action="store_true",
        help="Also save the raw binary prediction mask as a .npy file",
    )
    p.add_argument(
        "--no-colour", action="store_true",
        help="Disable ANSI colour codes in terminal output",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Input resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_mri_file(input_path: str) -> Tuple[Path, Optional[Path]]:
    """Return (mri_file, optional_gt_mask_file)."""
    p = Path(input_path)

    if p.is_file():
        if p.suffix in (".gz", ".nii") or ".nii" in p.name:
            gt_mask = next(p.parent.glob("*_gtv.nii.gz"), None)
            return p, gt_mask
        raise FileNotFoundError(f"Not a NIfTI file: {p}")

    if p.is_dir():
        # BraTS-style folder: contains *_t1c.nii.gz
        t1c = next(p.glob("*_t1c.nii.gz"), None)
        # IXI-style folder: contains a bare .nii file
        ixi = next(p.glob("*.nii"), None)
        mri_file = t1c or ixi
        if mri_file is None:
            raise FileNotFoundError(
                f"No MRI file found in {p}. "
                "Expected *_t1c.nii.gz (BraTS) or *.nii (IXI)."
            )
        gt_mask = next(p.glob("*_gtv.nii.gz"), None) if t1c else None
        return mri_file, gt_mask

    raise FileNotFoundError(f"Input does not exist: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    architecture_override: Optional[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, str, dict]:
    cp = Path(checkpoint_path)
    if not cp.is_absolute():
        cp = PROJECT_ROOT / cp
    if not cp.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {cp}\n"
            "Train the model first with:  python train.py --epochs 50"
        )
    state = torch.load(str(cp), map_location=device)
    arch = architecture_override or state.get("architecture", "segresnet")
    model = get_model(architecture=arch, in_channels=1, out_channels=2)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info("Loaded checkpoint: %s  (arch=%s, best_dice=%.4f)",
                cp.name, arch, state.get("best_dice", 0.0))
    return model, arch, state


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    spatial_size: Tuple[int, int, int],
    threshold: float,
    min_component_voxels: int,
) -> Dict[str, Any]:
    """
    Run full inference and return a rich results dictionary.

    Returns
    -------
    {
      "tumour_prob_map"   : np.ndarray (D, H, W) – per-voxel tumour probability at ORIGINAL resolution
      "prediction_mask"   : np.ndarray (D, H, W) – binary {0,1} mask at ORIGINAL resolution
      "max_probability"   : float  – peak tumour probability anywhere in volume
      "mean_probability"  : float  – mean tumour probability in top-1% of voxels
      "tumour_confidence" : float  – scalar confidence score [0, 1]
      "has_tumour"        : bool
      "tumour_voxels"     : int    – number of positive voxels in prediction
      "total_voxels"      : int
      "tumour_fraction"   : float  – tumour_voxels / total_voxels
      "tumour_volume_cc"  : float  – tumour volume in cm³ (assumes 1 mm³/voxel)
      "num_components"    : int    – number of distinct tumour components found
      "component_sizes"   : list[int]
    }
    """
    preprocessor = Preprocessor(spatial_size=spatial_size)
    sample = {
        "image": image_np.astype(np.float32),
        "mask": np.zeros_like(image_np, dtype=np.float32),
        "patient_id": "",
    }
    image_tensor = preprocessor(sample)["image"]   # (1, D, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 1, D, H, W)

    with torch.no_grad():
        logits = model(image_tensor)                      # (1, 2, D, H, W)
        probs = torch.softmax(logits, dim=1)[:, 1]        # (1, D, H, W) tumour channel

    # Up-sample probability map back to original MRI resolution
    original_shape = image_np.shape
    probs_orig = F.interpolate(
        probs.unsqueeze(0).float(),
        size=original_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)                               # (D, H, W)

    prob_np = probs_orig.cpu().numpy()

    # ── Binary mask ──────────────────────────────────────────────────────────
    binary = (prob_np > threshold).astype(np.uint8)

    # Remove noise with morphological opening + small component filtering
    if binary.any():
        binary = ndimage.binary_opening(binary, structure=np.ones((3, 3, 3))).astype(np.uint8)
        labels, num_labels = ndimage.label(binary)
        comp_sizes = np.bincount(labels.ravel())[1:]   # skip background label 0
        final_mask = np.zeros_like(binary, dtype=np.uint8)
        kept_sizes = []
        for lab_id, sz in enumerate(comp_sizes, start=1):
            if sz >= min_component_voxels:
                final_mask[labels == lab_id] = 1
                kept_sizes.append(int(sz))
        prediction_mask = final_mask
        num_components = len(kept_sizes)
    else:
        prediction_mask = binary
        num_components = 0
        kept_sizes = []

    # ── Scalar confidence metrics ────────────────────────────────────────────
    tumour_voxels = int(prediction_mask.sum())
    total_voxels = int(prediction_mask.size)
    tumour_fraction = tumour_voxels / total_voxels

    max_prob = float(prob_np.max())
    # "mean confidence": average probability in the predicted tumour region
    if tumour_voxels > 0:
        mean_prob_in_tumour = float(prob_np[prediction_mask == 1].mean())
    else:
        mean_prob_in_tumour = float(prob_np.max())   # fallback: peak prob

    # Top-1% of voxels (robustly captures hotspot even when no threshold hit)
    top1_threshold = np.percentile(prob_np, 99.0)
    mean_top1_prob = float(prob_np[prob_np >= top1_threshold].mean())

    # Overall confidence: combination of max probability and region coverage
    tumour_confidence = float(np.clip(max_prob, 0.0, 1.0))
    has_tumour = tumour_voxels > 0

    # Volume in cm³ assuming standard 1 mm isotropic spacing → 1 mm³ per voxel
    tumour_volume_cc = tumour_voxels * 1e-3  # mm³ → cm³

    return {
        "tumour_prob_map":      prob_np,
        "prediction_mask":      prediction_mask,
        "max_probability":      max_prob,
        "mean_probability_in_tumour": mean_prob_in_tumour,
        "mean_top1pct_probability":   mean_top1_prob,
        "tumour_confidence":    tumour_confidence,
        "has_tumour":           has_tumour,
        "tumour_voxels":        tumour_voxels,
        "total_voxels":         total_voxels,
        "tumour_fraction":      tumour_fraction,
        "tumour_volume_cc":     tumour_volume_cc,
        "num_components":       num_components,
        "component_sizes":      kept_sizes,
        "threshold_used":       threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth comparison (when a GTV mask is available)
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_gt(
    prediction_mask: np.ndarray,
    gt_mask_path: Path,
) -> Dict[str, float]:
    gt = (nib.load(str(gt_mask_path)).get_fdata(dtype=np.float32) > 0).astype(np.float32)
    pred = prediction_mask.astype(np.float32)

    # Ensure same shape (resize gt to pred if needed)
    if gt.shape != pred.shape:
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        gt_t = F.interpolate(gt_t, size=pred.shape, mode="nearest")
        gt = gt_t.squeeze().numpy()

    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())

    dice  = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou   = tp / (tp + fp + fn + 1e-8)
    prec  = tp / (tp + fp + 1e-8)
    rec   = tp / (tp + fn + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    acc   = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    spec  = tn / (tn + fp + 1e-8)
    gt_vol_cc = float(gt.sum()) * 1e-3

    return {
        "dice":            round(dice,  6),
        "iou":             round(iou,   6),
        "precision":       round(prec,  6),
        "recall":          round(rec,   6),
        "f1":              round(f1,    6),
        "pixel_accuracy":  round(acc,   6),
        "specificity":     round(spec,  6),
        "gt_tumour_voxels": int(gt.sum()),
        "gt_tumour_volume_cc": round(gt_vol_cc, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overlay visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_overlay(
    image_np: np.ndarray,
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    patient_id: str,
    output_dir: Path,
    has_tumour: bool,
) -> Path:
    """Save a 3-panel figure: central slice of MRI | probability heatmap | prediction overlay."""
    # Pick the slice with the most tumour voxels (or the central slice for healthy)
    if pred_mask.any():
        slice_idx = int(pred_mask.sum(axis=(1, 2)).argmax())
    else:
        slice_idx = image_np.shape[0] // 2

    mri_slice  = image_np[slice_idx]
    prob_slice = prob_map[slice_idx]
    mask_slice = pred_mask[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0e0e0e")

    verdict_color = "#ff4444" if has_tumour else "#44cc44"
    verdict_text  = "⚠ TUMOUR DETECTED" if has_tumour else "✓ NO TUMOUR"
    fig.suptitle(
        f"{patient_id}  —  {verdict_text}",
        color=verdict_color, fontsize=16, fontweight="bold",
    )

    # Panel 1: raw MRI
    axes[0].imshow(mri_slice.T, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("MRI (T1c)", color="white")
    axes[0].axis("off")

    # Panel 2: tumour probability heatmap
    im = axes[1].imshow(prob_slice.T, cmap="hot", origin="lower",
                        vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title("Tumour Probability", color="white")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    plt.getp(plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis, "ticklabels")

    # Panel 3: MRI + prediction overlay
    axes[2].imshow(mri_slice.T, cmap="gray", origin="lower", aspect="auto")
    if mask_slice.any():
        overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        axes[2].imshow(overlay, cmap="Reds", alpha=0.6, origin="lower",
                       vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Prediction Overlay", color="white")
    axes[2].axis("off")

    for ax in axes:
        ax.set_facecolor("#0e0e0e")

    out_path = output_dir / "overlays" / f"{patient_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Terminal report
# ─────────────────────────────────────────────────────────────────────────────

def _c(text: str, code: str, use_colour: bool) -> str:
    if not use_colour:
        return text
    RESET = "\033[0m"
    return f"\033[{code}m{text}{RESET}"


def print_report(
    patient_id: str,
    result: Dict[str, Any],
    gt_metrics: Optional[Dict[str, float]],
    checkpoint_info: Dict[str, Any],
    use_colour: bool = True,
) -> None:
    has_tumour    = result["has_tumour"]
    verdict_text  = "TUMOUR DETECTED" if has_tumour else "NO TUMOUR DETECTED"
    verdict_colour = "1;31" if has_tumour else "1;32"  # bold red / bold green

    sep = "═" * 60
    print()
    print(_c(sep, "36", use_colour))
    print(_c(f"  🧠  BRAIN TUMOUR DETECTION REPORT", "1;37", use_colour))
    print(_c(sep, "36", use_colour))
    print(f"  Patient ID   : {_c(patient_id, '1;37', use_colour)}")
    print(f"  Architecture : {checkpoint_info.get('architecture', 'segresnet')}")
    print(f"  Best Dice    : {checkpoint_info.get('best_dice', 0.0):.4f}  (from training)")
    print(_c(sep, "36", use_colour))

    # ── VERDICT ──────────────────────────────────────────────────────────────
    pad = " " * 18
    print()
    print(pad + _c(f"◉  {verdict_text}", verdict_colour, use_colour))
    print()

    # ── Confidence & probability ──────────────────────────────────────────────
    print(_c("  ── Confidence Scores ─────────────────────────────────", "33", use_colour))
    conf = result["tumour_confidence"]
    bar_len = 30
    filled = int(conf * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    bar_colour = "31" if conf >= 0.5 else "32"
    print(f"  Tumour Confidence    : {_c(bar, bar_colour, use_colour)} {conf * 100:.1f}%")
    print(f"  Max Voxel Prob.      : {result['max_probability']:.4f}")
    print(f"  Mean Prob (tumour)   : {result['mean_probability_in_tumour']:.4f}")
    print(f"  Mean Prob (top 1%)   : {result['mean_top1pct_probability']:.4f}")
    print(f"  Threshold Used       : {result['threshold_used']:.2f}")

    # ── Volume & size ─────────────────────────────────────────────────────────
    print(_c("  ── Volume Metrics ────────────────────────────────────", "33", use_colour))
    print(f"  Tumour Voxels        : {result['tumour_voxels']:,}")
    print(f"  Total Brain Voxels   : {result['total_voxels']:,}")
    print(f"  Tumour Fraction      : {result['tumour_fraction'] * 100:.4f}%")
    print(f"  Est. Tumour Volume   : {result['tumour_volume_cc']:.3f} cm³")
    print(f"  # Connected Regions  : {result['num_components']}")
    if result["component_sizes"]:
        sizes_str = ", ".join(f"{s:,}" for s in sorted(result["component_sizes"], reverse=True))
        print(f"  Component Sizes      : {sizes_str} voxels")

    # ── Ground-truth comparison (if GTV available) ────────────────────────────
    if gt_metrics:
        print(_c("  ── Ground-Truth Comparison (GTV mask found) ──────────", "33", use_colour))
        print(f"  GT Tumour Voxels     : {gt_metrics['gt_tumour_voxels']:,}")
        print(f"  GT Tumour Volume     : {gt_metrics['gt_tumour_volume_cc']:.3f} cm³")
        dice_str = f"{gt_metrics['dice']:.4f}"
        dice_colour = "1;32" if gt_metrics["dice"] >= 0.5 else "1;31"
        print(f"  Dice Score           : {_c(dice_str, dice_colour, use_colour)}")
        print(f"  IoU  Score           : {gt_metrics['iou']:.4f}")
        print(f"  Precision            : {gt_metrics['precision']:.4f}")
        print(f"  Recall               : {gt_metrics['recall']:.4f}")
        print(f"  F1  Score            : {gt_metrics['f1']:.4f}")
        print(f"  Pixel Accuracy       : {gt_metrics['pixel_accuracy']:.4f}")
        print(f"  Specificity          : {gt_metrics['specificity']:.4f}")
        print(f"  TP / FP / FN / TN    : {gt_metrics['tp']:,} / {gt_metrics['fp']:,} / {gt_metrics['fn']:,} / {gt_metrics['tn']:,}")

    print(_c(sep, "36", use_colour))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────────────────────

def save_json_report(
    patient_id: str,
    result: Dict[str, Any],
    gt_metrics: Optional[Dict[str, float]],
    checkpoint_info: Dict[str, Any],
    output_dir: Path,
) -> Path:
    report = {
        "patient_id":           patient_id,
        "verdict":              "TUMOUR" if result["has_tumour"] else "HEALTHY",
        "has_tumour":           result["has_tumour"],
        "confidence": {
            "tumour_confidence":          round(result["tumour_confidence"], 6),
            "max_voxel_probability":      round(result["max_probability"], 6),
            "mean_prob_in_tumour_region": round(result["mean_probability_in_tumour"], 6),
            "mean_prob_top1pct":          round(result["mean_top1pct_probability"], 6),
            "threshold_used":             result["threshold_used"],
        },
        "volume": {
            "tumour_voxels":    result["tumour_voxels"],
            "total_voxels":     result["total_voxels"],
            "tumour_fraction":  round(result["tumour_fraction"], 8),
            "tumour_volume_cc": round(result["tumour_volume_cc"], 4),
        },
        "morphology": {
            "num_components":  result["num_components"],
            "component_sizes": result["component_sizes"],
        },
        "model": {
            "architecture": checkpoint_info.get("architecture", "segresnet"),
            "best_dice":    checkpoint_info.get("best_dice", None),
            "epoch":        checkpoint_info.get("epoch", None),
        },
    }
    if gt_metrics:
        report["ground_truth_comparison"] = gt_metrics

    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_path = json_dir / f"{patient_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_logging("WARNING")  # Keep terminal clean; we print our own report

    device = get_device()

    # ── Resolve input ─────────────────────────────────────────────────────────
    try:
        mri_file, gt_mask_file = resolve_mri_file(args.input)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)

    patient_id = args.patient_id or mri_file.parent.name
    LOGGER.info("MRI file: %s", mri_file)
    LOGGER.info("GT mask : %s", gt_mask_file or "None")

    # ── Load image ────────────────────────────────────────────────────────────
    image_np = nib.load(str(mri_file)).get_fdata(dtype=np.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, arch, ckpt_state = load_model(
            args.checkpoint, args.architecture, device
        )
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)

    checkpoint_info = {
        "architecture": arch,
        "best_dice": ckpt_state.get("best_dice", None),
        "epoch": ckpt_state.get("epoch", None),
    }

    # ── Run inference ─────────────────────────────────────────────────────────
    result = run_inference(
        model=model,
        image_np=image_np,
        device=device,
        spatial_size=tuple(args.spatial_size),
        threshold=args.threshold,
        min_component_voxels=args.min_component_voxels,
    )

    # ── GT comparison ─────────────────────────────────────────────────────────
    gt_metrics: Optional[Dict[str, float]] = None
    if gt_mask_file is not None:
        gt_metrics = compare_with_gt(result["prediction_mask"], gt_mask_file)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    ensure_directory(out_dir / "overlays")
    ensure_directory(out_dir / "json")

    # ── Save overlay ──────────────────────────────────────────────────────────
    overlay_path = save_overlay(
        image_np=image_np,
        prob_map=result["tumour_prob_map"],
        pred_mask=result["prediction_mask"],
        patient_id=patient_id,
        output_dir=out_dir,
        has_tumour=result["has_tumour"],
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = save_json_report(
        patient_id=patient_id,
        result=result,
        gt_metrics=gt_metrics,
        checkpoint_info=checkpoint_info,
        output_dir=out_dir,
    )

    # ── Optionally save raw mask ───────────────────────────────────────────────
    if args.save_prediction:
        pred_dir = ensure_directory(out_dir / "predictions")
        import numpy as _np
        _np.save(pred_dir / f"{patient_id}.npy", result["prediction_mask"])

    # ── Print report to terminal ──────────────────────────────────────────────
    print_report(
        patient_id=patient_id,
        result=result,
        gt_metrics=gt_metrics,
        checkpoint_info=checkpoint_info,
        use_colour=not args.no_colour,
    )

    print(f"  Overlay saved  → {overlay_path}")
    print(f"  JSON report    → {json_path}")
    print()


if __name__ == "__main__":
    main()
