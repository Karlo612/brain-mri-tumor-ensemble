"""
Evaluate fine-tuned models and save confusion-matrix PNGs.
Usage:
    python src/eval.py --cfg src/config.yaml
"""

import argparse, yaml, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from utils       import set_global_determinism
from datamodule  import DataModule

# ----------------------------------------------------------------------
def evaluate_one(model_path: Path, dm: DataModule, plot_dir: Path) -> None:
    if not model_path.exists():
        print(f"⚠️  {model_path.name} not found — skipping")
        return

    print(f"\n🧪 Evaluating {model_path.name}")
    model = tf.keras.models.load_model(model_path)

    # quick overall metrics
    loss, acc, prec, rec = model.evaluate(dm.test_ds, verbose=0)
    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    # fresh iterator so we don’t reuse an exhausted dataset
    y_prob, y_true = [], []
    for batch_x, batch_y in dm.test_ds:
        y_prob.append(model(batch_x, training=False).numpy())
        y_true.append(batch_y.numpy())

    y_prob = np.vstack(y_prob)
    y_true = np.vstack(y_true)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_true, axis=1)

    print(classification_report(
        y_true, y_pred, target_names=dm.class_names, digits=4))

    # confusion-matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dm.class_names, yticklabels=dm.class_names)
    plt.title(f"Confusion — {model_path.stem}")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()

    plot_path = plot_dir / f"{model_path.stem}_cm.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  → saved confusion matrix to {plot_path}")

# ----------------------------------------------------------------------
def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_global_determinism(cfg["seed"])
    dm = DataModule(cfg_path)

    plot_dir  = Path(cfg["plot_dir"]);  plot_dir.mkdir(exist_ok=True)
    model_dir = Path(cfg["model_dir"])

    for bb in cfg["backbones"]:
        evaluate_one(model_dir / f"{bb}_finetuned.keras", dm, plot_dir)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="src/config.yaml")
    args = ap.parse_args()
    main(args.cfg)
