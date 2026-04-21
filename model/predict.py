"""
model/predict.py
----------------
Inference pipeline: takes a preprocessed NumPy array, runs it through the
ResNet-50 model, and returns a human-readable prediction with per-class
probabilities loaded dynamically from class_mapping.json.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Class labels ───────────────────────────────────────────────────────────────

_CLASS_MAPPING_PATH = "saved_models/class_mapping.json"

# Short → Full label map (matches class_mapping.json short names)
_SHORT_TO_FULL = {
    "No DR":          "No Diabetic Retinopathy",
    "Mild":           "Mild Diabetic Retinopathy",
    "Moderate":       "Moderate Diabetic Retinopathy",
    "Severe":         "Severe Diabetic Retinopathy",
    "Proliferative":  "Proliferative Diabetic Retinopathy",
}


def load_class_labels(mapping_path: str = _CLASS_MAPPING_PATH) -> List[str]:
    """
    Load DR class labels (full names) from class_mapping.json.

    The JSON maps string indices to short names, e.g.:
        {"0": "No DR", "1": "Mild", ...}

    Returns:
        List of full label strings ordered by class index.
    """
    if not os.path.exists(mapping_path):
        # Fallback to hardcoded defaults if file is missing
        return list(_SHORT_TO_FULL.values())

    with open(mapping_path, "r") as f:
        raw: Dict[str, str] = json.load(f)

    # Sort by integer key so the list is in correct class order
    ordered = [raw[str(i)] for i in range(len(raw))]

    # Convert short names to full names where possible
    return [_SHORT_TO_FULL.get(short, short) for short in ordered]


# Module-level labels — loaded once at import time
DR_LABELS: List[str] = load_class_labels()


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(
    model: nn.Module,
    image_array: np.ndarray,
) -> Tuple[str, Dict[str, float]]:
    """
    Run inference on a single preprocessed image.

    Args:
        model:        A PyTorch model (ResNet-50 or compatible) in eval mode.
        image_array:  NumPy float32 array of shape (1, 3, 224, 224),
                      with ImageNet normalization already applied.

    Returns:
        predicted_label:  Full string label for the top-1 class.
        probabilities:    Dict mapping each full label → probability (0-1).
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
    probs_numpy = probs_tensor.squeeze(0).cpu().numpy()

    # 5. Top prediction
    predicted_index = int(np.argmax(probs_numpy))
    predicted_label = DR_LABELS[predicted_index]

    # 6. Build probability dictionary
    probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(DR_LABELS, probs_numpy)
    }

    return predicted_label, probabilities
