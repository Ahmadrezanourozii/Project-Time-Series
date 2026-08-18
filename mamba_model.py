"""Mamba (State Space Model) classifier for windowed VGRF sequences.

Uses the official `mamba_ssm` package (CUDA/Triton kernels) when available
-- the Kaggle GPU path -- and falls back to the pure-PyTorch reference block
in mamba_minimal.py otherwise (local smoke tests only). Follows the same
interface conventions as ResNet1DClassifier / RecurrentClassifier: input
(B, C, T), output (B, num_classes) logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba, Mamba2

    HAVE_MAMBA_SSM = True
except ImportError:
    from mamba_minimal import MambaBlockRef

    HAVE_MAMBA_SSM = False

# d_model * expand must stay divisible by Mamba2's headdim (64), so only
# these presets are exposed.
MODEL_SIZES = {
    "small": {"d_model": 64, "n_layers": 2},
    "base": {"d_model": 128, "n_layers": 4},
    "large": {"d_model": 256, "n_layers": 6},
}


def _build_mixer(variant: str, d_model: int, d_state: int, expand: int) -> nn.Module:
    if not HAVE_MAMBA_SSM:
        return MambaBlockRef(d_model, d_state=min(d_state, 16), expand=expand)
    if variant == "mamba2":
        return Mamba2(d_model=d_model, d_state=d_state, expand=expand)
    if variant == "mamba1":
        return Mamba(d_model=d_model, d_state=min(d_state, 16), expand=expand)
    raise ValueError(f"Unknown variant: {variant}")


class MambaClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.1,
        variant: str = "mamba2",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if (d_model * expand) % 64 != 0:
            raise ValueError(f"d_model*expand={d_model * expand} must be divisible by 64 (Mamba2 headdim)")
        self.grad_checkpoint = grad_checkpoint

        self.in_proj = nn.Linear(in_channels, d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.mixers = nn.ModuleList(
            _build_mixer(variant, d_model, d_state, expand) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.in_proj(x)
        for norm, mixer in zip(self.norms, self.mixers):
            if self.grad_checkpoint and self.training:
                x = x + checkpoint(lambda inp, m=mixer, n=norm: m(n(inp)), x, use_reentrant=False)
            else:
                x = x + mixer(norm(x))
        x = self.final_norm(x).mean(dim=1)  # mean-pool over time
        return self.head(self.dropout(x))
