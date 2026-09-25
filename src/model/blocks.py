"""Reusable convolutional building blocks for the U-Net family."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """``(conv 3x3 -> BN -> ReLU) x 2``, the standard U-Net unit.

    Parameters
    ----------
    in_channels, out_channels:
        Input and output channel counts.
    mid_channels:
        Channels between the two convolutions; defaults to *out_channels*.
        The decoder uses ``in_channels // 2`` here for the bilinear variant.
    dropout:
        Optional 2-D dropout probability applied after the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling stage: max-pool by 2 then :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def pad_to_match(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Pad *x* symmetrically so its spatial size matches *reference*.

    Odd input sizes make the pooled feature maps lose a pixel, so the upsampled
    decoder tensor can be one pixel smaller than its skip connection. Padding
    (rather than cropping the skip) keeps the output the same size as the
    input, which matters for inference on arbitrary user images.
    """
    diff_y = reference.size(-2) - x.size(-2)
    diff_x = reference.size(-1) - x.size(-1)
    if diff_y == 0 and diff_x == 0:
        return x
    if diff_y < 0 or diff_x < 0:
        # The decoder overshot: centre-crop back down to the reference size.
        top = max(0, -diff_y) // 2
        left = max(0, -diff_x) // 2
        return x[..., top : top + reference.size(-2), left : left + reference.size(-1)]
    return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])


class Up(nn.Module):
    """Upscaling stage: upsample, concatenate the skip connection, convolve.

    Parameters
    ----------
    in_channels:
        Channels of the *upsampled* tensor before concatenation.
    skip_channels:
        Channels of the encoder feature map being concatenated.
    out_channels:
        Channels produced by the block.
    bilinear:
        ``True`` uses parameter-free bilinear upsampling (fewer parameters,
        no checkerboard artefacts); ``False`` uses a transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels // 2 + skip_channels
        self.conv = DoubleConv(conv_in, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = pad_to_match(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """Final 1x1 convolution mapping features to class logits."""

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def init_weights(module: nn.Module) -> None:
    """Kaiming-initialise convolutions and reset norm layers.

    Apply with ``model.apply(init_weights)``. Pretrained encoders should be
    initialised *before* their weights are loaded, or skipped entirely.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
