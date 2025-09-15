"""
Grad-CAM utilities with automatic last-conv-layer detection.
Usage example (in a notebook):

    from pathlib import Path
    import tensorflow as tf
    from src.datamodule import DataModule
    from src.gradcam import save_gradcam_grid

    PROJECT_ROOT = Path("/content/drive/MyDrive/Final_2025_Comp/SP")
    dm = DataModule(str(PROJECT_ROOT / "src/config.yaml"))

    model = tf.keras.models.load_model(PROJECT_ROOT / "models" / "efficientnetb0_finetune.keras")
    save_gradcam_grid(model, dm, model_name="efficientnetb0_finetune", num_samples=8)

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def find_last_conv_layer(model):
    """
    Walk backwards through layers and return the name of the last 2D conv-like layer.
    Handles Conv2D, SeparableConv2D, DepthwiseConv2D.
    Raises if none found.
    """
    conv_types = (tf.keras.layers.Conv2D,
                  tf.keras.layers.SeparableConv2D,
                  tf.keras.layers.DepthwiseConv2D)
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
    raise RuntimeError("No 2D convolutional layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a single (1, H, W, 3) image batch.
    If last_conv_layer_name is None, detect it automatically.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Model mapping input -> (last conv outputs, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradients of the top predicted class wrt conv layer
    grads = tape.gradient(class_channel, conv_out)

    # Global average pooling the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]  # HxWxC
    heatmap = tf.tensordot(conv_out, pooled_grads, axes=[2, 0])

    # ReLU & normalise 0-1
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, img_uint8, alpha=0.4):
    """
    Resize heatmap to image size, colour it, and overlay onto the original image.
    img_uint8 expected shape (H, W, 3) in [0,255].
    """
    h, w = img_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, img_uint8, 1 - alpha, 0)
    return overlay


# --------------------------------------------------------------
# Main public API
# --------------------------------------------------------------
def save_gradcam_grid(model, datamodule, model_name, num_samples=8, save_dir=None):
    """
    Create & save a grid of Grad-CAM overlays for a few test images.

    Parameters
    ----------
    model : tf.keras.Model
    datamodule : DataModule (needs .test_ds and .class_names)
    model_name : str  -> used in figure title & filename
    num_samples : int -> number of images to visualise (max drawn from first batch)
    save_dir : pathlib.Path or str or None
        Directory to save PNG. Defaults to PROJECT_ROOT/plots if None.

    Returns
    -------
    Path to saved PNG.
    """
    if save_dir is None:
        # assume same structure as elsewhere in project
        save_dir = Path("/content/drive/MyDrive/Final_2025_Comp/SP/plots")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Try to find last conv automatically once
    try:
        last_conv = find_last_conv_layer(model)
    except RuntimeError as e:
        print("Grad-CAM skipped:", e)
        return None

    # Grab one batch from test
    test_iter = iter(datamodule.test_ds)
    x_batch, y_batch = next(test_iter)

    max_show = min(num_samples, x_batch.shape[0])
    h, w = model.input_shape[1:3]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i in range(max_show):
        img = x_batch[i:i+1]                          # (1,H,W,3) float32 0-255
        true_label = int(tf.argmax(y_batch[i]).numpy())

        # Prediction
        preds = model.predict(img, verbose=0)
        pred_label = int(np.argmax(preds[0]))
        pred_conf  = float(preds[0][pred_label])

        # Heatmap
        try:
            heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name=last_conv)
            # Ensure image is uint8
            img_uint8 = img[0].numpy().astype(np.uint8)
            overlay   = overlay_heatmap_on_image(heatmap, img_uint8, alpha=0.4)

            axes[i].imshow(overlay)
            axes[i].set_title(
                f"True: {datamodule.class_names[true_label]}\n"
                f"Pred: {datamodule.class_names[pred_label]} ({pred_conf:.2%})",
                color=("green" if true_label == pred_label else "red")
            )
            axes[i].axis("off")

        except Exception as ex:
            axes[i].imshow(img[0].numpy().astype(np.uint8))
            axes[i].set_title(f"Grad-CAM failed\n{ex}", color="red", fontsize=8)
            axes[i].axis("off")
            print(f"Grad-CAM failed for sample {i}: {ex}")

    # Fill unused axes (if any)
    for j in range(max_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=16)
    plt.tight_layout()
    out_path = save_dir / f"{model_name}_gradcam.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved Grad-CAM grid to {out_path}")
    return out_path
