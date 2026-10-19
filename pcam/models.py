"""Models. Each returns one logit per patch (shape [batch]); sigmoid(logit) = P(tumour)."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from pcam.config import ModelConfig


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU, then halve the spatial size with max-pooling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # BN has its own bias
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A compact CNN trained from scratch: 96x96 -> 48 -> 24 -> 12 -> 6, then pool."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # average each channel -> 256 numbers per patch
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x).squeeze(1)


class ResNet18Binary(nn.Module):
    """torchvision ResNet18 with its 1000-class head replaced by a single logit."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "small_cnn":
        return SmallCNN()
    if cfg.name == "resnet18":
        return ResNet18Binary(pretrained=cfg.pretrained)
    raise ValueError(f"Unknown model: {cfg.name}")  # unreachable: config validates names


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
