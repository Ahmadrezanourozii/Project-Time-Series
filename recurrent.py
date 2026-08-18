from __future__ import annotations

import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        cell_type: str,
        in_channels: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        cell = cell_type.lower()
        if cell == "lstm":
            rnn_cls = nn.LSTM
        elif cell == "gru":
            rnn_cls = nn.GRU
        elif cell == "rnn":
            rnn_cls = nn.RNN
        else:
            raise ValueError(f"Unsupported recurrent cell type: {cell_type}")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        out_features = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, T, C]
        seq = x.transpose(1, 2)
        out, _ = self.encoder(seq)
        pooled = out.mean(dim=1)
        pooled = self.drop(pooled)
        return self.head(pooled)
