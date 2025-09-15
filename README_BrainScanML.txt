BrainScanML – Stratified Brain MRI Tumour Classification with Ensembles & Calibration
Author: Karlo Nahro (ID: 19003070)
Project code: 6G6Z0019 Synoptic Project – Creative Piece
Notebook: Nahro_Karlo_19003070_BrainScanML_synoptic.ipynb

1. What this does (in one breath)
Classifies axial brain MRI slices into four classes: glioma, meningioma, pituitary, notumor, using three transfer‑learned CNN backbones (Xception, VGG16, EfficientNetB0), a mean‑probability ensemble, and temperature scaling to calibrate the probabilities.
Held‑out test set: Accuracy 0.9756, Macro‑F1 0.9749, Expected Calibration Error (ECE) 0.0153.

Why it works (first principles):
- Stratified split preserves class ratios → fairer, lower‑variance metrics.
- Transfer learning reuses low-level visual features → less data, faster convergence.
- Ensembling reduces model idiosyncrasies by averaging predictions.
- Calibration adjusts softmax so 80% confidence ≈ 80% correct — vital if a human relies on the score.

2. Quickstart (examiner‑proof)
1) Create & activate an environment (change tool as you like)
    conda create -n brainscanml python=3.11 -y
    conda activate brainscanml
2) Install dependencies
    pip install -r requirements.txt
3) Put the dataset here (see §3)
    data/
      raw_brainMRI/
        glioma/
        meningioma/
        pituitary/
        notumor/
4) (First run only) create stratified splits
    python utils/make_split.py --data_dir data/raw_brainMRI --out_dir data/dataset_brain_split
5) Reproduce results (or just open the notebook)
    jupyter notebook Nahro_Karlo_19003070_BrainScanML_synoptic.ipynb
    # or run the CLI wrappers:
    Google Colab (one‑click): https://colab.research.google.com/drive/14ICaazpXRV67IdvICp4sQpFtlySQ5XHb?usp=sharing

    python src/train.py --cfg src/config.yaml --backbone all
    python src/evaluate.py --cfg src/config.yaml --calibrate

Outputs (models, metrics, plots) drop into outputs/.

3. Project layout (what’s in the ZIP)
.
├── Nahro_Karlo_19003070_BrainScanML_synoptic.ipynb
├── requirements.txt
├── src/
│   datamodule.py
│   models.py
│   train.py
│   evaluate.py
│   utils.py
├── utils/
│   make_split.py
├── data/                      (empty in the ZIP; you add the dataset)
│   raw_brainMRI/              original class folders
│   dataset_brain_split/       auto-created {train,val,test}/class/... + splits.csv
└── outputs/
    checkpoints/
    figures/                   confusion matrix, ROC, reliability diagrams, Grad-CAMs
    test_metrics.csv
    per_class_report.csv

4. Data details
Source: Public brain MRI tumour dataset (four-class version commonly seen on Kaggle).
Structure expected: data/raw_brainMRI/<class>/<image>.jpg|.png (mixed extensions handled).
Split: ~75% train / 13% val / 12% test (stratified).
Re-splitting: Only if you change the dataset; otherwise keep splits.csv for reproducibility.

5. Reproducing the headline numbers
Train each backbone:
    python src/train.py --cfg src/config.yaml --backbone xception
    python src/train.py --cfg src/config.yaml --backbone vgg16
    python src/train.py --cfg src/config.yaml --backbone efficientnetb0
Ensemble + Calibration:
    python src/evaluate.py --cfg src/config.yaml --calibrate
Creates:
    test_metrics_final.csv (accuracy, precision/recall/F1, ECE, Brier, AUROC)
    figures/reliability_diagram.png, roc_curves.png, confusion_matrix.png

6. Configuration & determinism
Key knobs in src/config.yaml (or the “Config” cell in the notebook):
    seed: 42
    img_size: 224
    batch_size: 32
    epochs_head: 5
    epochs_finetune: 15
    backbones: [xception, vgg16, efficientnetb0]
    paths: data_dir, splits_csv, out_dir
set_global_determinism(42) fixes seeds for Python, NumPy and TensorFlow and requests deterministic ops where possible.

7. Dependencies
Pinned in requirements.txt:
    tensorflow>=2.14
    scikit-learn>=1.3
    opencv-python
    pandas
    numpy
    matplotlib
    pyyaml
Plus Jupyter for interactive runs. (Google Colab cells included but optional.)

8. Licence & academic integrity
Dataset licence: See original provider.
Code: © Karlo Nahro (academic use only under MMU assessment rules).
Generative AI tools were not used to create the artefact, in line with MMU policy.

9. How to mark this (dear examiner 👋)
This README satisfies the Creative Piece requirement to include instructions on how to use the artefact and the required software (e.g. Python interpreter).

10. Contact
Email: Karlo.Nahro@stu.mmu.ac.uk
EthOS Reference Number: 76551
