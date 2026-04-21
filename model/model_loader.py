"""
model/model_loader.py
---------------------
Defines the model architecture and helpers to load weights.

Two loaders are provided:
  - load_resnet_model()  → uses the real ResNet-50 backbone (dr_model_resnet50.pth)
  - load_model()         → legacy dummy-model loader (kept for compatibility)
"""

import os
import torch
import torch.nn as nn
from torchvision import models


# ── Number of DR severity classes ─────────────────────────────────────────────
NUM_CLASSES = 5   # No DR | Mild | Moderate | Severe | Proliferative DR


# ── ResNet-50 model ────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build a ResNet-50 model with the final fully-connected layer replaced
    to output `num_classes` logits.

    Args:
        num_classes: Number of output classes (default: 5).

    Returns:
        ResNet-50 nn.Module with the fc layer adapted for DR classification.
    """
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_resnet_model(
    weights_path: str = "saved_models/dr_model_resnet50.pth",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Instantiate a ResNet-50 model and load the provided weights.

    Args:
        weights_path: Path to the .pth weights file saved with
                      ``torch.save(model.state_dict(), path)``.
        num_classes:  Number of output classes (default: 5).

    Returns:
        ResNet-50 model in eval mode on CPU.

    Raises:
        FileNotFoundError: If `weights_path` does not exist.
    """
    model = build_resnet50(num_classes=num_classes)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[model_loader] ❌ Weights not found at '{weights_path}'. "
            "Please place 'dr_model_resnet50.pth' in the saved_models/ directory."
        )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[model_loader] OK Loaded ResNet-50 weights from '{weights_path}'")

    model.eval()
    return model


# -- Legacy dummy model ---------------------------------------------------------

# Flattened size of a 224x224 RGB image  (3 x 224 x 224)
INPUT_FEATURES = 3 * 224 * 224


class DummyDRModel(nn.Module):
    """
    Minimal dummy model kept for backward compatibility.
    NOT used in production inference.
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
        x = self.flatten(x)
        return self.classifier(x)


def load_model(weights_path: str = "saved_models/model.pth") -> DummyDRModel:
    """Legacy dummy-model loader. Use `load_resnet_model` for real inference."""
    model = DummyDRModel(num_classes=NUM_CLASSES)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[model_loader] OK Loaded dummy weights from '{weights_path}'")
    else:
        print(
            f"[model_loader] WARNING No weights found at '{weights_path}'. "
            "Using randomly initialised weights (dummy mode)."
        )
    model.eval()
    return model
