from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + residual)
        return out


class ResNet1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(base_channels, base_channels, blocks=3, stride=1)
        self.layer2 = self._make_stage(base_channels, base_channels * 2, blocks=4, stride=2)
        self.layer3 = self._make_stage(base_channels * 2, base_channels * 4, blocks=6, stride=2)
        self.layer4 = self._make_stage(base_channels * 4, base_channels * 8, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(base_channels * 8, num_classes)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [ResidualBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).squeeze(-1)
        out = self.drop(out)
        return self.head(out)
