"""Pure-PyTorch Mamba-1-style block (reference selective scan).

Used automatically by mamba_model.MambaClassifier when the CUDA/Triton
`mamba_ssm` package is not importable (local macOS smoke tests). The scan is
a sequential Python loop over timesteps -- correct but slow; it exists so the
full pipeline can be exercised end-to-end on CPU/MPS before pushing to
Kaggle, not for real training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MambaBlockRef(nn.Module):
    """Input/output: (B, T, d_model)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_in = F.silu(x_in)

        y = self._selective_scan(x_in)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        B, T, d_inner = u.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        dbc = self.x_proj(u)  # (B, T, dt_rank + 2*d_state)
        dt, B_mat, C_mat = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)

        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, T, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * B_mat.unsqueeze(2) * u.unsqueeze(-1)

        h = torch.zeros(B, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C_mat[:, t]))
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        return y + u * self.D
