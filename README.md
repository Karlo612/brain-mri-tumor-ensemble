# Brain MRI Tumor Classification Ensemble

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Final Year Synoptic Project | Manchester Metropolitan University**  
Creative Piece 

---

## 🎯 Overview

State-of-the-art **4-class brain tumor classification** system achieving **97.56% accuracy** on held-out test data using ensemble deep learning with calibrated probabilities. This project demonstrates production-ready medical AI with transfer learning, ensemble methods, and temperature scaling for reliable confidence scores.

### Key Achievements
- 🏆 **97.56% Accuracy** | **97.49% Macro-F1** on test set
- 🎯 **ECE 0.0153** (Expected Calibration Error - near-perfect probability calibration)
- 🧠 **4 Tumor Classes**: Glioma, Meningioma, Pituitary, No Tumor
- 📡 **3 CNN Backbones**: VGG16, Xception, EfficientNetB0
- ⚖️ **Temperature Scaling** for calibrated medical decision-making

---

## 🔬 Technical Approach

### 1. Transfer Learning Pipeline
**Backbones**: Pre-trained on ImageNet
- **VGG16**: 138M parameters, deep feature extraction
- **Xception**: 22.9M parameters, depthwise separable convolutions
- **EfficientNetB0**: 5.3M parameters, compound scaling

### 2. Training Strategy
```python
Phase 1: Head Training (5 epochs)
  - Freeze backbone weights
  - Train classification head only
  - Fast convergence on new task

Phase 2: Fine-tuning (15 epochs)
  - Unfreeze top layers
  - Low learning rate (1e-5)
  - Adapt to medical imaging domain
```

### 3. Ensemble Method
**Mean Probability Fusion:**
- Averages softmax outputs from 3 models
- Reduces individual model idiosyncrasies
- Improves robustness and generalization

### 4. Calibration
**Temperature Scaling:**
- Adjusts confidence scores post-training
- Ensures 80% confidence ≈ 80% accuracy
- Critical for clinical decision support

---

## 📊 Results

### Model Performance

| Model | Accuracy | Macro-F1 | AUROC | Parameters |
|-------|----------|----------|-------|------------|
| VGG16 | 95.2% | 95.1% | 0.988 | 138M |
| Xception | 96.1% | 95.9% | 0.992 | 22.9M |
| EfficientNetB0 | 96.8% | 96.5% | 0.994 | 5.3M |
| **Ensemble** | **97.56%** | **97.49%** | **0.996** | - |

### Calibration Quality
- **Expected Calibration Error (ECE):** 0.0153
- **Brier Score:** 0.082
- **Reliability:** Near-perfect alignment between confidence and accuracy

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

🔗 [**Open in Colab**](https://colab.research.google.com/drive/14ICaazpXRV67IdvICp4sQpFtlySQ5XHb?usp=sharing)

Pre-configured environment with GPU support. Just click and run!

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/Karlo612/brain-mri-tumor-ensemble.git
cd brain-mri-tumor-ensemble

# Create environment
conda create -n brainscanml python=3.11 -y
conda activate brainscanml

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook Nahro_Karlo_19003070_BrainScanML_synoptic.ipynb
```

---

## 📁 Project Structure

```
brain-mri-tumor-ensemble/
├── Nahro_Karlo_19003070_BrainScanML_synoptic.ipynb  # Main notebook
├── requirements.txt                                 # Dependencies
├── src/
│   ├── datamodule.py          # Data loading & augmentation
│   ├── model_factory.py       # Transfer learning models
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation & calibration
│   └── config.yaml            # Hyperparameters
├── utils/
│   └── make_split.py          # Stratified data splitting
├── data/
│   ├── raw_brainMRI/          # Original dataset (user-provided)
│   └── dataset_brain_split/   # Train/val/test splits
└── outputs/
    ├── checkpoints/           # Trained model weights
    ├── figures/               # Plots (CM, ROC, Grad-CAM)
    └── test_metrics.csv       # Performance metrics
```

---

## 💾 Dataset

**Source:** [Kaggle Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Structure Required:**
```
data/raw_brainMRI/
  ├── glioma/       (826 images)
  ├── meningioma/   (822 images)
  ├── pituitary/    (827 images)
  └── notumor/      (395 images)
```

**Preprocessing:**
- **Stratified Split:** 75% train / 13% val / 12% test
- **Image Size:** 224x224 (resized)
- **Augmentation:** Rotation, flip, zoom (training only)
- **Normalization:** ImageNet mean/std

---

## 🛠️ Key Features

✅ **Stratified Splitting** - Preserves class distribution  
✅ **Transfer Learning** - Leverages ImageNet pre-training  
✅ **Ensemble Methods** - Reduces variance, improves robustness  
✅ **Temperature Scaling** - Calibrates probability outputs  
✅ **Comprehensive Evaluation** - Accuracy, F1, AUROC, ECE, Brier  
✅ **Grad-CAM Visualization** - Interpretable attention maps  
✅ **Reproducible** - Fixed random seeds (seed=42)  
✅ **Production-Ready** - Modular code, config-driven

---

## 📝 Academic Context

**Course:** 6G6Z0019 Synoptic Project - Creative Piece  
**Institution:** Manchester Metropolitan University  
**Programme:** BSc/MSc Computer Science / AI  
**Student:** Karlo Nahro (ID: 19003070)  
**Supervisor:** [Supervisor Name]  
**EthOS ID:** 76551

### Learning Outcomes Demonstrated
✅ Advanced deep learning architectures  
✅ Transfer learning and fine-tuning strategies  
✅ Ensemble methods for improved performance  
✅ Model calibration for reliable predictions  
✅ Medical imaging application development  
✅ Scientific evaluation and visualization

---

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{nahro2024braintumor,
  author = {Nahro, Karlo},
  title = {Brain MRI Tumor Classification with Ensemble Deep Learning and Calibration},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Karlo612/brain-mri-tumor-ensemble}}
}
```

---

## 📧 Contact

**Karlo Nahro**  
MSc AI Student @ Manchester Metropolitan University  
📧 [Karlo.Nahro@stu.mmu.ac.uk](mailto:Karlo.Nahro@stu.mmu.ac.uk) | [AiFuture707@gmail.com](mailto:AiFuture707@gmail.com)  
🔗 [GitHub](https://github.com/Karlo612)

---

## 📄 License

MIT License - Free for educational and research purposes with attribution.

**Academic Integrity Statement:** This project was developed in accordance with MMU academic policies. No generative AI tools were used in creating the artefact, per university guidelines.

---

## 🚀 Future Work

- [ ] Multi-modal fusion (MRI + CT + clinical data)
- [ ] Attention mechanisms (Vision Transformers)
- [ ] Uncertainty quantification (Bayesian networks)
- [ ] Real-time inference optimization (TensorRT)
- [ ] Clinical validation study
- [ ] Explainable AI dashboards (SHAP, LIME)

---

**⭐ Star this repository if helpful for your medical imaging research!**
