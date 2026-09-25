"""U-Net for binary crack segmentation, written from scratch.

Follows Ronneberger et al. (2015) with batch normalisation, ``same`` padding
and a configurable depth/width so the same code covers a small CPU-friendly
model and a full-size one.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .blocks import DoubleConv, Down, OutConv, Up, init_weights

LOGGER = logging.getLogger(__name__)


class UNet(nn.Module):
    """Encoder/decoder U-Net producing per-pixel logits.

    Parameters
    ----------
    in_channels:
        Input channels (3 for RGB).
    num_classes:
        Output channels. ``1`` for binary segmentation with
        :class:`~torch.nn.BCEWithLogitsLoss`.
    base_channels:
        Channel count of the first encoder stage; each stage doubles it.
    depth:
        Number of downsampling stages. With ``depth=4`` the network downsamples
        by 16, so inputs should ideally be multiples of 16 -- odd sizes still
        work because the decoder pads to match its skip connections.
    bilinear:
        Use bilinear upsampling instead of transposed convolutions.
    dropout:
        Dropout2d probability applied at the bottleneck and decoder stages.

    Notes
    -----
    ``forward`` returns **logits**. Apply ``torch.sigmoid`` yourself when you
    need probabilities; the losses in :mod:`src.losses` expect logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        **_ignored,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.bilinear = bilinear

        encoder_channels = [base_channels * (2**i) for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, encoder_channels[0])
        self.downs = nn.ModuleList(
            [
                Down(encoder_channels[i], encoder_channels[i + 1], dropout=dropout if i == depth - 1 else 0.0)
                for i in range(depth)
            ]
        )

        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                Up(
                    in_channels=encoder_channels[i],
                    skip_channels=encoder_channels[i - 1],
                    out_channels=encoder_channels[i - 1],
                    bilinear=bilinear,
                    dropout=dropout,
                )
            )

        self.head = OutConv(encoder_channels[0], num_classes)
        self.apply(init_weights)
        LOGGER.debug(
            "Built UNet(depth=%d, base_channels=%d, bilinear=%s)", depth, base_channels, bilinear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(N, C, H, W)`` inputs to ``(N, num_classes, H, W)`` logits."""
        if x.dim() != 4:
            raise ValueError(f"expected a 4-D (N, C, H, W) tensor, got {tuple(x.shape)}")
        skips: list[torch.Tensor] = []
        feat = self.stem(x)
        for down in self.downs:
            skips.append(feat)
            feat = down(feat)
        for up, skip in zip(self.ups, reversed(skips), strict=False):
            feat = up(feat, skip)
        return self.head(feat)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
