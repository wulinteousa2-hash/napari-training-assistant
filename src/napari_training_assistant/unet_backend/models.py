from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block_2d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _conv_block_3d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block_2d(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _conv_block_2d(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _conv_block_2d(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _conv_block_2d(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block_2d(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block_2d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block_2d(base * 2, base)

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = _conv_block_3d(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _conv_block_3d(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = _conv_block_3d(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = _conv_block_3d(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block_3d(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block_3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block_3d(base * 2, base)

        self.head = nn.Conv3d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)