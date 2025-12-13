"""Data loading pipeline with deterministic splits and validation."""

from pathlib import Path
from dataclasses import dataclass
import yaml
import tensorflow as tf


@dataclass
class DataModule:
    cfg_path: str

    def __post_init__(self):
        self.cfg_path = Path(self.cfg_path)
        with open(self.cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self._build()

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else (self.cfg_path.parent / path).resolve()

    def _build(self):
        root = self._resolve_path(self.cfg["data_dir"])
        img_size = tuple(self.cfg["img_size"])
        bs = self.cfg["batch_size"]
        seed = self.cfg["seed"]
        class_mode = self.cfg["class_mode"]

        if not root.exists():
            raise FileNotFoundError(
                f"data_dir {root} not found. Provide a dataset with train/val/test folders or"
                " update config.yaml to point to your dataset root."
            )

        train_dir = root / "train"
        val_dir = root / "val"
        test_dir = root / "test"

        if train_dir.exists() and val_dir.exists():
            self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                train_dir, labels="inferred", label_mode=class_mode,
                image_size=img_size, batch_size=bs, shuffle=True, seed=seed
            )
            self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                val_dir, labels="inferred", label_mode=class_mode,
                image_size=img_size, batch_size=bs, shuffle=False
            )
        else:
            print("⚠️ No explicit train/val folders found. Falling back to validation_split.")
            val_split = self.cfg["val_split"]
            self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                root, labels="inferred", label_mode=class_mode,
                validation_split=val_split, subset="training", seed=seed,
                image_size=img_size, batch_size=bs
            )
            self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                root, labels="inferred", label_mode=class_mode,
                validation_split=val_split, subset="validation", seed=seed,
                image_size=img_size, batch_size=bs
            )

        if test_dir.exists():
            self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_dir, labels="inferred", label_mode=class_mode,
                image_size=img_size, batch_size=bs, shuffle=False
            )
        else:
            print("⚠️  No test/ folder found – using validation split as stand-in test.")
            self.test_ds = self.val_ds

        self._class_names = self.train_ds.class_names

        autotune = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(autotune)
        self.val_ds = self.val_ds.prefetch(autotune)
        self.test_ds = self.test_ds.prefetch(autotune)

    @property
    def class_names(self):
        return self._class_names
