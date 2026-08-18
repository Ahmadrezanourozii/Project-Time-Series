"""Model factory wrapping the existing (unmodified) architectures.

ResNet1DClassifier and RecurrentClassifier live at the repo root as
pre-existing, already-designed classes -- they are imported and used as-is
here. Only `in_channels` (9/9/18, foot-config dependent) and `dropout` (an
existing constructor argument on both classes) vary; every other
constructor default (base_channels, block counts, hidden_size, num_layers,
bidirectional) is left untouched.
"""

from __future__ import annotations

from torch import nn

from recurrent import RecurrentClassifier
from resnet1d import ResNet1DClassifier

NUM_CLASSES = 2


def build_model(model_name: str, in_channels: int, dropout: float, **model_kwargs) -> nn.Module:
    if model_name == "resnet":
        return ResNet1DClassifier(in_channels=in_channels, num_classes=NUM_CLASSES, dropout=dropout)
    if model_name == "gru":
        return RecurrentClassifier(
            cell_type="gru",
            in_channels=in_channels,
            num_classes=NUM_CLASSES,
            dropout=dropout,
        )
    if model_name == "mamba":
        from mamba_model import MambaClassifier

        return MambaClassifier(
            in_channels=in_channels,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            **model_kwargs,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
