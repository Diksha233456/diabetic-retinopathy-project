"""
utils/gradcam.py
----------------
Placeholder for Grad-CAM (Gradient-weighted Class Activation Mapping).

Grad-CAM produces a heatmap highlighting the regions of the retinal image
that most influenced the model's prediction — useful for clinical explainability.

TODO (future milestone):
    - Hook into the last convolutional layer of the backbone.
    - Compute gradients of the target class score w.r.t. feature maps.
    - Weight feature maps by their averaged gradients.
    - Overlay the resulting heatmap on the original image.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391
"""

# ── Coming soon ────────────────────────────────────────────────────────────────


def generate_gradcam(model, image_tensor, target_class: int):
    """
    Generate a Grad-CAM heatmap for the given image and target class.

    Args:
        model:         Trained PyTorch model (must expose conv layers).
        image_tensor:  Preprocessed input tensor of shape (1, 3, 224, 224).
        target_class:  Integer index of the class to explain (0–4).

    Returns:
        heatmap: 2-D NumPy array (H, W) normalized to [0, 1].

    Raises:
        NotImplementedError: Until this module is implemented.
    """
    raise NotImplementedError(
        "Grad-CAM is not yet implemented. "
        "See the docstring above for the planned approach."
    )
