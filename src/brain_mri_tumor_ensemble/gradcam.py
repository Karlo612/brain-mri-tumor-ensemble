"""
Grad-CAM utilities with automatic last-conv-layer detection.

Example CLI:
    python -m brain_mri_tumor_ensemble.gradcam \
        --cfg config.yaml \
        --model_path outputs/models/xception_finetune.keras
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from brain_mri_tumor_ensemble.datamodule import DataModule


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def find_last_conv_layer(model):
    """Return the name of the last 2D convolutional layer."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
    raise RuntimeError("No 2D convolutional layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.tensordot(conv_out, pooled_grads, axes=[2, 0])
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, img_uint8, alpha=0.4):
    h, w = img_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, img_uint8, 1 - alpha, 0)
    return overlay


# --------------------------------------------------------------
# Main public API
# --------------------------------------------------------------
def save_gradcam_grid(model, datamodule, model_name, num_samples=8, save_dir=None):
    """Create & save a grid of Grad-CAM overlays for a few test images."""
    if save_dir is None:
        save_dir = Path("outputs/plots")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        last_conv = find_last_conv_layer(model)
    except RuntimeError as e:
        print("Grad-CAM skipped:", e)
        return None

    test_iter = iter(datamodule.test_ds)
    x_batch, y_batch = next(test_iter)

    max_show = min(num_samples, x_batch.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i in range(max_show):
        img = x_batch[i:i + 1]
        true_label = int(tf.argmax(y_batch[i]).numpy())

        preds = model.predict(img, verbose=0)
        pred_label = int(np.argmax(preds[0]))
        pred_conf = float(preds[0][pred_label])

        try:
            heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name=last_conv)
            img_uint8 = img[0].numpy().astype(np.uint8)
            overlay = overlay_heatmap_on_image(heatmap, img_uint8, alpha=0.4)

            axes[i].imshow(overlay)
            axes[i].set_title(
                f"True: {datamodule.class_names[true_label]}\n"
                f"Pred: {datamodule.class_names[pred_label]} ({pred_conf:.2%})",
                color=("green" if true_label == pred_label else "red")
            )
            axes[i].axis("off")

        except Exception as ex:  # pragma: no cover - visual debug path
            axes[i].imshow(img[0].numpy().astype(np.uint8))
            axes[i].set_title(f"Grad-CAM failed\n{ex}", color="red", fontsize=8)
            axes[i].axis("off")
            print(f"Grad-CAM failed for sample {i}: {ex}")

    for j in range(max_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=16)
    plt.tight_layout()
    out_path = save_dir / f"{model_name}_gradcam.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM grid to {out_path}")
    return out_path


def cli():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM grid for a trained model.")
    parser.add_argument("--cfg", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_path", required=True, help="Path to a trained Keras model")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to visualise")
    parser.add_argument("--out_dir", default=None, help="Optional output directory for Grad-CAM grid")
    args = parser.parse_args()

    dm = DataModule(args.cfg)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    save_gradcam_grid(model, dm, model_name=model_path.stem,
                      num_samples=args.num_samples, save_dir=args.out_dir)


if __name__ == "__main__":
    cli()
