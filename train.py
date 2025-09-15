import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import argparse, yaml
from pathlib import Path

import tensorflow as tf  # keep this at top-level

from src.utils import set_global_determinism
from src.datamodule import DataModule
from src.model_factory import build_model, compile_model


def unfreeze_backbone_layers(model, n_last: int):
    """Unfreeze last n_last layers of the conv backbone (before GAP)."""
    # Find GAP layer index (start of dense head)
    gap_idx = None
    for i, lyr in enumerate(model.layers):
        if isinstance(lyr, tf.keras.layers.GlobalAveragePooling2D):
            gap_idx = i
            break
    if gap_idx is None:
        raise RuntimeError("Could not locate GlobalAveragePooling2D layer.")

    backbone_layers = model.layers[:gap_idx]  # everything before GAP
    # Unfreeze last n_last conv layers (skip BatchNorms)
    for lyr in backbone_layers[-n_last:]:
        if not isinstance(lyr, tf.keras.layers.BatchNormalization):
            lyr.trainable = True
    # Keep BN frozen (often stabilises fine-tuning)
    for lyr in backbone_layers:
        if isinstance(lyr, tf.keras.layers.BatchNormalization):
            lyr.trainable = False


def main(cfg_path: str, backbone: str):
    # ---------- load config ----------
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_global_determinism(cfg["seed"])
    dm = DataModule(cfg_path)
    img_size     = tuple(cfg["img_size"])
    num_classes  = len(dm.class_names)

    model_dir = Path(cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1 : head training ----------
    model = build_model(backbone, img_size, num_classes, base_trainable=False)
    compile_model(model, lr=cfg["lr_head"])

    ckpt_head = model_dir / f"{backbone}_head.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stop_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_head),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(cfg["log_dir"]) / f"{backbone}_head")
        )
    ]

    model.fit(
        dm.train_ds,
        validation_data=dm.val_ds,
        epochs=cfg["epochs_head"],
        callbacks=callbacks,
        verbose=1
    )

    # ---------- Phase 2 : fine-tune ----------
    unfreeze_backbone_layers(model, cfg["unfreeze_layers"])
    compile_model(model, lr=cfg["lr_finetune"])

    ckpt_ft = model_dir / f"{backbone}_finetune.keras"
    callbacks[1].filepath = str(ckpt_ft)
    callbacks[2].log_dir  = str(Path(cfg["log_dir"]) / f"{backbone}_finetune")

    model.fit(
        dm.train_ds,
        validation_data=dm.val_ds,
        epochs=cfg["epochs_finetune"],
        callbacks=callbacks,
        verbose=1
    )

    print(f"✅ Finished training {backbone}. Head → {ckpt_head.name}, fine-tuned → {ckpt_ft.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",      default="src/config.yaml")
    ap.add_argument("--backbone", choices=["xception","vgg16","efficientnetb0"],
                    default="xception")
    args = ap.parse_args()
    main(args.cfg, args.backbone)
