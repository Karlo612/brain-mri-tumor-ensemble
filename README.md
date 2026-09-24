# Brain MRI tumour classification

I built BrainScanML for my **6G6Z0019 Synoptic Project in July 2025**. It compares Xception, VGG16 and EfficientNetB0 on four MRI image categories: glioma, meningioma, pituitary and no tumour. I combined their predictions by averaging probabilities and used Grad-CAM and reliability plots to explore the results.

## Saved project results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Ensemble | 97.56% | 97.49% |
| EfficientNetB0 | 96.62% | 96.52% |
| Xception | 96.04% | 95.95% |
| VGG16 | 96.04% | 95.95% |

These are historical results from `test_metrics_final.csv`, not a new run. `splits.csv` records 5,349 training, 945 validation and 859 test images. The split is stratified by image class, not grouped by patient. No patient-independent or clinical performance is claimed.

The saved ensemble ECE is approximately 0.01528. This is confidence evaluation of the averaged probabilities, **not evidence of fitted temperature scaling**.

## Files

- `Nahro_Karlo_BrainScanML_synoptic.ipynb`: restored submitted notebook with saved outputs and corrected, shorter explanations. Its code retains the original Colab paths and has not been rerun after restoration.
- `train.py`, `eval.py`, `src/brain_mri_tumor_ensemble/`: maintained local code.
- `test_metrics_final.csv`, `splits.csv`, repository PNGs: historical results and split manifest.
- `Nahro_Karlo_BrainScanML_Report.pdf`: historical submitted report; read the corrections in `SUBMISSION_NOTES.md` alongside it.

## Run locally

Install Python 3.9+ and run `pip install -e .`. Provide the dataset as `data/dataset_brain_split/{train,val,test}/<class>/` and adjust `config.yaml` if needed. All three split folders are required. Dataset images and trained checkpoints are not bundled.

```sh
python train.py --cfg config.yaml --backbone xception
python train.py --cfg config.yaml --backbone vgg16
python train.py --cfg config.yaml --backbone efficientnetb0
python eval.py --cfg config.yaml
```

The evaluation script requires all configured checkpoints, evaluates them on the same test batches and calculates the mean-probability ensemble. It writes accuracy, macro-F1 and ECE to `outputs/plots/test_metrics_current.csv`, plus confusion matrices. It does not fit temperature scaling or overwrite the historical results. Library versions and hardware may affect reproducibility.

## Data and limitations

The submitted notes cite [tombackert's Brain Tumor MRI Data](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data). The exact downloaded snapshot and its upstream sources remain unverified; a similarly named collection should not be assumed equivalent. The split manifest does not establish patient separation or absence of duplicate image content.

This is a student research project. Grad-CAM is a qualitative diagnostic, and neither the accuracy nor ECE establishes clinical usefulness. I would next clarify dataset provenance, introduce patient-grouped evaluation where possible and assess transfer to an independent dataset.
