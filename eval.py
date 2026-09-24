"""Evaluate all configured checkpoints and their mean-probability ensemble."""
import argparse
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from brain_mri_tumor_ensemble.datamodule import DataModule
from brain_mri_tumor_ensemble.utils import set_global_determinism


def expected_calibration_error(y, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    total = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def main(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    dm = DataModule(str(cfg_path))
    cfg = dm.cfg
    set_global_determinism(cfg['seed'])
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else cfg_path.parent / p
    model_dir, plot_dir = resolve(cfg['model_dir']), resolve(cfg['plot_dir'])
    paths = [model_dir / f'{name}_finetune.keras' for name in cfg['backbones']]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('All ensemble checkpoints are required: ' + ', '.join(missing))
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError('Configure at least two distinct backbones')
    models = [tf.keras.models.load_model(p, compile=False) for p in paths]
    truth, per_model = [], [[] for _ in models]
    # A single dataset traversal guarantees identical images/order for all models.
    for images, labels in dm.test_ds:
        truth.append(labels.numpy().argmax(axis=1))
        for model, rows in zip(models, per_model):
            rows.append(model(images, training=False).numpy())
    y = np.concatenate(truth)
    predictions = [np.concatenate(rows) for rows in per_model]
    for probabilities in predictions:
        if probabilities.shape != (len(y), len(dm.class_names)) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid prediction shape or nonfinite probabilities')
        if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Expected class probabilities')
    predictions.append(np.mean(predictions, axis=0))
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, probabilities in zip(cfg['backbones'] + ['ensemble'], predictions):
        predicted = probabilities.argmax(axis=1)
        rows.append(dict(model=name, acc=float(accuracy_score(y, predicted)),
                         f1_macro=float(f1_score(y, predicted, labels=range(len(dm.class_names)), average='macro', zero_division=0)),
                         ECE=expected_calibration_error(y, probabilities), n_test_images=len(y)))
        cm = confusion_matrix(y, predicted, labels=range(len(dm.class_names)))
        ConfusionMatrixDisplay(cm, display_labels=dm.class_names).plot(xticks_rotation=45)
        plt.tight_layout()
        plt.savefig(plot_dir / f'{name}_cm.png', dpi=200)
        plt.close()
    destination = plot_dir / 'test_metrics_current.csv'
    with destination.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved new evaluation to {destination}; historical metrics were not overwritten.')
    for row in rows:
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config.yaml')
    main(parser.parse_args().cfg)
