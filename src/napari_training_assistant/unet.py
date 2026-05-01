"""Basic U-Net architecture helpers.

The plugin stores architecture metadata even before a full trainer is wired in.
This module gives the default ``basic_unet`` backend a concrete meaning and can
be used by the future training runner.
"""

from __future__ import annotations

from typing import Any


def describe_basic_unet(architecture: dict[str, Any]) -> dict[str, Any]:
    """Return a concise architecture summary for the configured basic U-Net."""

    depth = int(architecture.get("depth", 4))
    base_channels = int(architecture.get("base_channels", 32))
    output_mode = architecture.get("output_mode", "binary")
    num_classes = int(architecture.get("num_classes", 2))
    output_channels = 1 if output_mode == "binary" else num_classes
    feature_channels = [base_channels * (2**index) for index in range(depth + 1)]
    return {
        "backend": "basic_unet",
        "spatial_dims": architecture.get("spatial_dims", "2d"),
        "preset": architecture.get("preset", "standard_unet"),
        "depth": depth,
        "base_channels": base_channels,
        "feature_channels": feature_channels,
        "double_conv_blocks": depth + 1 + depth,
        "conv_layers_per_block": 2,
        "double_conv_layers": (depth + 1 + depth) * 2,
        "upsampling_layers": depth,
        "final_projection_layers": 1,
        "input_channels": int(architecture.get("input_channels", 1)),
        "output_channels": output_channels,
        "output_mode": output_mode,
    }


def build_basic_unet_2d(architecture: dict[str, Any]):
    """Build a PyTorch 2D U-Net for the configured ``basic_unet`` backend."""

    import torch
    from torch import nn

    if architecture.get("spatial_dims", "2d") != "2d":
        raise ValueError("Only 2D basic U-Net is implemented.")
    if architecture.get("preset", "standard_unet") != "standard_unet":
        raise ValueError("Only the standard U-Net preset is implemented.")

    normalization = architecture.get("normalization", "batch")
    upsampling = architecture.get("upsampling", "transpose")
    depth = int(architecture.get("depth", 4))
    base_channels = int(architecture.get("base_channels", 32))
    input_channels = int(architecture.get("input_channels", 1))
    output_channels = 1 if architecture.get("output_mode", "binary") == "binary" else int(
        architecture.get("num_classes", 2)
    )

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                _normalization_layer(nn, normalization, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                _normalization_layer(nn, normalization, out_channels),
                nn.ReLU(inplace=True),
            ]
            self.block = nn.Sequential(*[layer for layer in layers if layer is not None])

        def forward(self, x):
            return self.block(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            if upsampling == "transpose":
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                conv_in_channels = out_channels + skip_channels
            elif upsampling == "bilinear":
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                )
                conv_in_channels = out_channels + skip_channels
            else:
                raise ValueError(f"Unsupported upsampling mode: {upsampling}")
            self.conv = DoubleConv(conv_in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat([skip, x], dim=1))

    class BasicUNet2D(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [base_channels * (2**index) for index in range(depth + 1)]
            self.encoders = nn.ModuleList()
            in_channels = input_channels
            for out_channels in channels[:-1]:
                self.encoders.append(DoubleConv(in_channels, out_channels))
                in_channels = out_channels
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottleneck = DoubleConv(channels[-2], channels[-1])
            decoder_blocks = []
            in_channels = channels[-1]
            for skip_channels in reversed(channels[:-1]):
                decoder_blocks.append(UpBlock(in_channels, skip_channels, skip_channels))
                in_channels = skip_channels
            self.decoders = nn.ModuleList(decoder_blocks)
            self.head = nn.Conv2d(base_channels, output_channels, kernel_size=1)

        def forward(self, x):
            skips = []
            for encoder in self.encoders:
                x = encoder(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for decoder, skip in zip(self.decoders, reversed(skips)):
                x = decoder(x, skip)
            return self.head(x)

    return BasicUNet2D()


def _normalization_layer(nn, mode: str, channels: int):
    if mode == "batch":
        return nn.BatchNorm2d(channels)
    if mode == "instance":
        return nn.InstanceNorm2d(channels)
    if mode == "group":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if mode == "none":
        return None
    raise ValueError(f"Unsupported normalization mode: {mode}")
