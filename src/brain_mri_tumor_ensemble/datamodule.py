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

        missing = [str(p) for p in (train_dir, val_dir, test_dir) if not p.is_dir()]
        if missing:
            raise FileNotFoundError("Separate train, val and test folders are required: " + ", ".join(missing))
        names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        for folder in (val_dir, test_dir):
            if sorted(p.name for p in folder.iterdir() if p.is_dir()) != names:
                raise ValueError("Class folders differ across splits")
        def load(folder, shuffle):
            return tf.keras.preprocessing.image_dataset_from_directory(
                folder, labels="inferred", label_mode=class_mode, class_names=names,
                image_size=img_size, batch_size=bs, shuffle=shuffle, seed=seed)
        self.train_ds = load(train_dir, True)
        self.val_ds = load(val_dir, False)
        self.test_ds = load(test_dir, False)

        self._class_names = self.train_ds.class_names

        autotune = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(autotune)
        self.val_ds = self.val_ds.prefetch(autotune)
        self.test_ds = self.test_ds.prefetch(autotune)

    @property
    def class_names(self):
        return self._class_names
