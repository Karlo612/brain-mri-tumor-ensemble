"""Utility helpers for determinism and reproducibility."""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_determinism(seed: int = 42) -> None:
    """Fix randomness for Python, NumPy and TensorFlow.

    Notes
    -----
    TensorFlow determinism still depends on the available kernels.
    For GPUs, ensure deterministic cuDNN kernels are installed and avoid
    unsupported ops to guarantee reproducible results.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Extra belt-and-braces: request deterministic kernels where possible.
    tf.config.experimental.enable_op_determinism()


if __name__ == "__main__":
    set_global_determinism(42)
    print("Random sanity check:")
    print("  python  :", random.random())
    print("  numpy   :", np.random.rand())
    print("  tf rand :", tf.random.uniform((1,)).numpy()[0])
