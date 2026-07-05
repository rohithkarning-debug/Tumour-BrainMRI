# Brain Tumour MRI Segmentation with MONAI

## Overview
This project builds a production-quality 3D segmentation pipeline for brain tumour MRI volumes using MONAI. It is designed for the BraTS 2024 MEN-RT dataset and supports training and inference from local NIfTI files.

## Features
- Reads paired T1 contrast MRI and GTV mask volumes from the BraTS dataset.
- Trains a 3D segmentation model with MONAI.
- Supports configurable model architectures via the `architecture` argument.
- Saves predicted tumour masks, overlay images, and JSON reports automatically.

## Model Architecture Justification
- SegResNet is the default architecture because it is robust for volumetric medical segmentation and offers a strong balance between accuracy, stability, and memory efficiency.
- UNet is available as an alternative for a simpler baseline and can be selected via configuration.
- The project is structured so future architectures such as SwinUNETR can be introduced without rewriting the training and inference entry points.

## Dataset
The project expects data under:
- input/BraTS-MEN-RT-Train-v2/<patient_id>/<patient_id>_t1c.nii.gz
- input/BraTS-MEN-RT-Train-v2/<patient_id>/<patient_id>_gtv.nii.gz

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python train.py --architecture segresnet --epochs 3
```

## Inference
```bash
python inference.py
```

## Results
The inference workflow writes outputs to:
- output/overlays/<patient_id>.png
- output/json/<patient_id>.json
- output/masks/<patient_id>.npy
- output/predictions/<patient_id>.npy

## Project Structure
- src/data_loader.py: loads MRI and mask volumes
- src/preprocess.py: defines normalization and augmentation transforms
- src/model.py: model factory for SegResNet and UNet
- src/trainer.py: training loop, checkpointing, metrics, and logging
- src/predictor.py: loads checkpoints and runs inference
- src/visualize.py: generates overlays
- src/export_json.py: writes JSON reports
- train.py: training entry point
- inference.py: inference entry point

## Future Work
- Add support for SwinUNETR and other transformer-based backbones.
- Expand to multi-modal inputs beyond T1c.
- Introduce patch-based training for larger and higher-resolution volumes.
