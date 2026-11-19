"""Residual 2D CNN for log-Mel speech deepfake detection."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention that reweights learned acoustic maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.layers(self.pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResidualBlock(nn.Module):
    """Two-convolution residual block; first convolution can downsample."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.attention = SqueezeExcitation(out_channels)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(self.attention(x) + residual)


class LogMelCNN(nn.Module):
    """Moderate-capacity residual CNN mapping [B, 1, mel, frames] to spoof logits."""

    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(24, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 96),
            ResidualBlock(96, 128),
        )
        # Mean and max pooling retain both broad spectral trends and salient cues.
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(inputs))
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.classifier(pooled).squeeze(1)
