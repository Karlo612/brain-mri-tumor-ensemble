# Brain MRI Tumor Classification Ensemble

Transfer learning ensemble (Xception, VGG16, EfficientNetB0) for 4-way brain tumor diagnosis with calibrated probabilities.

## Quick facts
- **Task:** Glioma vs. Meningioma vs. Pituitary vs. No Tumor
- **Best model:** Soft-voting ensemble of three CNNs
- **Reported test metrics:** Accuracy 97.56%, Macro-F1 97.49% (see `test_metrics_final.csv`)
- **Evidence artifacts:** Confusion/reliability/ROC plots in `outputs/plots/` (see repository PNGs) and Grad-CAM examples

## Repository layout
```
brain-mri-tumor-ensemble/
├── config.yaml                     # Default hyperparameters and paths (repo-relative)
├── train.py                        # Training entry point
├── eval.py                         # Evaluation entry point
├── Makefile                        # Convenience commands (install/train/eval/gradcam/safety scan)
├── src/brain_mri_tumor_ensemble/
│   ├── __init__.py
│   ├── datamodule.py               # Dataset construction and validation
│   ├── gradcam.py                  # Grad-CAM generation (also runnable as a module)
│   ├── model_factory.py            # Backbone registry and model builders
│   └── utils.py                    # Determinism utilities
├── test_metrics_final.csv          # Canonical test metrics table
├── splits.csv                      # Example dataset split manifest
├── Nahro_Karlo_BrainScanML_synoptic.ipynb
└── outputs/                        # Default output root (created on first run)
    ├── models/                     # Saved checkpoints (one per backbone)
    ├── logs/                       # TensorBoard logs
    └── plots/                      # Confusion matrices, ROC, reliability, Grad-CAM
```

## Installation
```bash
python -m pip install --upgrade pip
pip install -e .
```
Python 3.9+ is required. The `pyproject.toml` now declares the small runtime set needed to run the scripts; use `requirements.txt` if you prefer an explicit, minimal list of the same dependencies.

## Configuration
`config.yaml` now uses repository-relative defaults so it works out-of-the-box:
- `data_dir`: `data/dataset_brain_split` (expects `train/`, `val/`, `test/` subfolders)
- `model_dir`, `log_dir`, `plot_dir`: under `outputs/`

Relative paths are resolved against the location of the config file. Override any field on the CLI, for example:
```bash
python train.py --cfg path/to/your_config.yaml --backbone efficientnetb0
```

## Running the pipeline
- **Train a single backbone**
  ```bash
  make train BACKBONE=xception   # or vgg16 / efficientnetb0
  ```
- **Evaluate fine-tuned checkpoints and write confusion matrices**
  ```bash
  make eval
  ```
- **Generate Grad-CAM overlays for a checkpoint**
  ```bash
  make gradcam MODEL=outputs/models/xception_finetune.keras
  ```
- **Verify no personal university ID remains in the repo**
  ```bash
  make check_personal_ids
  ```

All commands assume checkpoints are stored in `outputs/models/` (default). Missing checkpoints now cause a clear error/exit code.

## Data expectations & methodology
- Download the Kaggle dataset locally and unpack it under `data/dataset_brain_split/` with `train/`, `val/`, and `test/` class subfolders.
- Dataset source: [Kaggle Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (not bundled).
- Directory layout: place class folders under `data/dataset_brain_split/train`, `val`, and `test`. If only a single folder is provided, the loader will fall back to `validation_split` using the configured `val_split` ratio.
- Preprocessing: images resized to 224×224; softmax head with categorical cross-entropy; per-backbone ImageNet preprocessing.
- Training regime: head training then fine-tuning (`epochs_head=15`, `epochs_finetune=20`, `lr_head=1e-3`, `lr_finetune=1e-4`, `early_stop_patience=4`, last 30 conv layers unfrozen during fine-tune).
- Reproducibility: `seed=42` is applied across Python, NumPy, and TensorFlow; deterministic ops are requested (`TF_DETERMINISTIC_OPS=1`). GPU determinism still depends on the installed kernels.

## Evidence for reported metrics
- Numerical results: `test_metrics_final.csv` stores the ensemble and backbone-level scores used in the report.
- Visual diagnostics: the repository includes the confusion matrix, reliability, and ROC plots produced from these checkpoints (`cm_test_ensemble.png`, `reliability_test_ensemble.png`, `roc_test_ensemble.png`). Run `make eval` to regenerate confusion matrices for your own checkpoints.
- Interpretability: Grad-CAM samples such as `xception_finetune_gradcam.png` are in `outputs/plots/`; regenerate via the Grad-CAM command above.

## Notes for extension
- The codebase is now packaged (`pip install -e .`) to stabilise imports across notebooks, scripts, and CI.
- CLI entry points prefer explicit `--cfg` arguments so alternative datasets or hyperparameters can be swapped without editing code.
