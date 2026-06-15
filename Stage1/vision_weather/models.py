from __future__ import annotations

import torch
import torch.nn as nn

class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + identity)
        return out


class TinyResNetEncoder(nn.Module):
    """轻量 CNN 编码器，适合中小数据量天气分类。"""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _ResidualBlock(width, width, stride=1)
        self.stage2 = _ResidualBlock(width, width * 2, stride=2)
        self.stage3 = _ResidualBlock(width * 2, width * 4, stride=2)
        self.stage4 = _ResidualBlock(width * 4, width * 4, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return x


class WeatherClassifier(nn.Module):
    """天气图像分类器（仅保留轻量 tiny_resnet）。"""

    def __init__(
        self,
        *,
        num_classes: int,
        dropout: float,
        resnet_width: int = 32,
    ) -> None:
        super().__init__()
        self.model_name = "tiny_resnet"
        self.encoder = TinyResNetEncoder(width=resnet_width)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_classes),
        )

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(pixel_values)
        logits = self.classifier(feat) # 输入图像经过网络输出 logits（每个类别一个分数）
        return logits
