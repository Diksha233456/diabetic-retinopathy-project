"""
model/predict.py
----------------
Inference pipeline: takes a preprocessed NumPy array, runs it through the
model, and returns a human-readable prediction with per-class probabilities.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model.model_loader import DummyDRModel


# ── Class labels (FULL FORMS) ─────────────────────────────────────────────────
DR_LABELS = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


def predict(
    model: DummyDRModel,
    image_array: np.ndarray,
) -> Tuple[str, Dict[str, float]]:
    """
    Run inference on a single preprocessed image.

    Args:
        model:        A DummyDRModel (or compatible model) already in eval mode.
        image_array:  NumPy float32 array of shape (1, 3, 224, 224)

    Returns:
        predicted_label:  String label for the top-1 class
        probabilities:    Dict mapping each label → probability
    """

    # 1. Convert NumPy → Tensor (ensure float32)
    tensor = torch.from_numpy(image_array).float()

    # 2. Move tensor to same device as model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # 3. Forward pass (no gradients)
    with torch.no_grad():
        logits = model(tensor)

    # 4. Softmax → probabilities
    probs_tensor = F.softmax(logits, dim=1)

    # Move back to CPU before numpy conversion
    probs_numpy = probs_tensor.squeeze(0).cpu().numpy()

    # 5. Get top prediction
    predicted_index = int(np.argmax(probs_numpy))
    predicted_label = DR_LABELS[predicted_index]

    # 6. Build probability dictionary
    probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(DR_LABELS, probs_numpy)
    }

    return predicted_label, probabilities
