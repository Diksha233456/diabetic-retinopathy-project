"""
model/model_loader.py
---------------------
Defines a lightweight dummy model and a helper to load it.

The DummyDRModel is intentionally simple — a single fully-connected layer —
so the project runs without GPU or heavy dependencies while you wire up the
rest of the pipeline.  Swap it with a real backbone (ResNet, EfficientNet …)
when you're ready for training.
"""

import os
import torch
import torch.nn as nn


# ── Number of DR severity classes ─────────────────────────────────────────────
NUM_CLASSES = 5   # No DR | Mild | Moderate | Severe | Proliferative DR

# Flattened size of a 224×224 RGB image  (3 × 224 × 224)
INPUT_FEATURES = 3 * 224 * 224


# ── Model definition ───────────────────────────────────────────────────────────

class DummyDRModel(nn.Module):
    """
    Minimal dummy model for Diabetic Retinopathy Detection.

    Architecture:
        Flatten → Linear(150_528 → 256) → ReLU → Linear(256 → 5)

    This is NOT a real medical model.  It produces random-ish outputs and is
    only meant to validate the end-to-end inference pipeline.

    Args:
        num_classes: Number of output classes (default: 5).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(INPUT_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 224, 224).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self.flatten(x)
        return self.classifier(x)


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_model(weights_path: str = "saved_models/model.pth") -> DummyDRModel:
    """
    Instantiate the model and optionally load saved weights.

    If `weights_path` exists the function loads the state-dict from that file;
    otherwise it logs a warning and returns the model with random weights.
    The model is always set to evaluation mode before being returned.

    Args:
        weights_path: Path to a .pth file produced by torch.save(model.state_dict(), …).

    Returns:
        model: DummyDRModel in eval mode, on CPU.

    Example:
        >>> model = load_model("saved_models/model.pth")
        >>> model
        DummyDRModel(...)
    """
    model = DummyDRModel(num_classes=NUM_CLASSES)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[model_loader] ✅ Loaded weights from '{weights_path}'")
    else:
        print(
            f"[model_loader] ⚠️  No weights found at '{weights_path}'. "
            "Using randomly initialised weights (dummy mode)."
        )

    model.eval()   # Disable dropout / batchnorm training behaviour
    return model
