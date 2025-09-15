import tensorflow as tf
from typing import Tuple, Dict, Callable

# --------------------------------------------------------------------
# 1)  Map backbone name  ->  (constructor, preprocess_input)
# --------------------------------------------------------------------
BACKBONES: Dict[str, Tuple[Callable, Callable]] = dict(
    xception       = (
        tf.keras.applications.Xception,
        tf.keras.applications.xception.preprocess_input
    ),
    vgg16          = (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input
    ),
    efficientnetb0 = (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input
    ),
)

# --------------------------------------------------------------------
# 2)  Build the transfer-learning model
# --------------------------------------------------------------------
def build_model(backbone: str,
                img_size: Tuple[int, int],
                num_classes: int,
                base_trainable: bool = False) -> tf.keras.Model:
    """
    backbone: one of 'xception', 'vgg16', 'efficientnetb0'
    img_size: (H, W)
    num_classes: number of output classes
    base_trainable: if True, leave the conv base unfrozen (for fine-tune phase)
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. "
                         f"Choose from {list(BACKBONES)}")

    Base, preprocess = BACKBONES[backbone]

    inputs  = tf.keras.Input(shape=(*img_size, 3))
    x       = preprocess(inputs)                     # match backbone expectations
    base    = Base(include_top=False,
                   weights="imagenet",
                   input_tensor=x)
    base.trainable = base_trainable                 # freeze or unfreeze

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs,
                          name=f"{backbone}_transfer")
# --------------------------------------------------------------------
# 3)  Standard compile helper
# --------------------------------------------------------------------
def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
    )
print("✓  model_factory.py written")
