import os, random, numpy as np, tensorflow as tf

def set_global_determinism(seed: int = 42) -> None:
    """
    Fixes randomness for Python, NumPy and TensorFlow, and requests
    deterministic GPU/CPU ops where possible.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"   # TF 2.12+ flag
    os.environ["PYTHONHASHSEED"]       = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Extra belt-and-braces: make cuDNN/eigen choose deterministic paths
    tf.config.experimental.enable_op_determinism()
    # NB: some ops are still nondeterministic

print("✓ utils.py written.")

# -------- smoke-test --------
if __name__ == "__main__":
    set_global_determinism(42)
    import tensorflow as tf, numpy as np
    print("Random sanity check:")
    print("  python  :", random.random())
    print("  numpy   :", np.random.rand())
    print("  tf rand :", tf.random.uniform((1,)).numpy()[0])
