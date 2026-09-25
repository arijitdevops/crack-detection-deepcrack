"""U-Net with a torchvision ResNet encoder.

The ResNet34 backbone gives the decoder ImageNet features, which converges
considerably faster than the from-scratch :class:`~src.model.unet.UNet` on a
dataset of a few hundred images.

Encoder taps (for a ``H x W`` input):

=========  ==================================  ========  =========
stage      module                              stride    channels
=========  ==================================  ========  =========
``x0``     ``conv1 + bn1 + relu``              /2        64
``x1``     ``maxpool + layer1``                /4        64
``x2``     ``layer2``                          /8        128
``x3``     ``layer3``                          /16       256
``x4``     ``layer4`` (bottleneck)             /32       512
=========  ==================================  ========  =========
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .blocks import DoubleConv, OutConv, Up, init_weights

LOGGER = logging.getLogger(__name__)


def _load_resnet34(pretrained: bool) -> nn.Module:
    """Instantiate torchvision ResNet34 using the modern ``weights=`` API.

    Falls back to random initialisation with a warning when the ImageNet
    weights cannot be downloaded (offline machine, proxy, no cache).
    """
    from torchvision.models import ResNet34_Weights, resnet34

    if not pretrained:
        return resnet34(weights=None)
    try:
        return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    except Exception as exc:  # network / cache failures surface in many shapes
        LOGGER.warning(
            "Could not load ImageNet weights for ResNet34 (%s); "
            "falling back to random initialisation",
            exc,
        )
        return resnet34(weights=None)


class UNetResNet34(nn.Module):
    """U-Net decoder on a ResNet34 encoder, returning per-pixel logits.

    Parameters
    ----------
    in_channels:
        Input channels. Values other than 3 replace ``conv1`` with a freshly
        initialised convolution (pretrained weights no longer apply to it).
    num_classes:
        Output channels; ``1`` for binary crack segmentation.
    pretrained:
        Load ImageNet weights into the encoder.
    decoder_channels:
        Output channels of the four skip-connected decoder stages, coarse to
        fine.
    bilinear:
        Bilinear upsampling (``True``) or transposed convolutions (``False``).
    freeze_encoder:
        Freeze all encoder parameters, e.g. for a quick linear-probe run.
    """

    ENCODER_CHANNELS: tuple[int, int, int, int, int] = (64, 64, 128, 256, 512)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 64),
        bilinear: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        **_ignored,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        encoder = _load_resnet34(pretrained)
        if in_channels != 3:
            encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            init_weights(encoder.conv1)

        # /2, 64 channels
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        # /4, 64 channels
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2  # /8,  128
        self.layer3 = encoder.layer3  # /16, 256
        self.layer4 = encoder.layer4  # /32, 512

        c0, c1, c2, c3, c4 = self.ENCODER_CHANNELS
        d3, d2, d1, d0 = decoder_channels

        self.up3 = Up(c4, c3, d3, bilinear=bilinear, dropout=dropout)  # /32 -> /16
        self.up2 = Up(d3, c2, d2, bilinear=bilinear, dropout=dropout)  # /16 -> /8
        self.up1 = Up(d2, c1, d1, bilinear=bilinear, dropout=dropout)  # /8  -> /4
        self.up0 = Up(d1, c0, d0, bilinear=bilinear, dropout=dropout)  # /4  -> /2

        # Final /2 -> /1 step has no encoder skip connection.
        self.final_up = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if bilinear
            else nn.ConvTranspose2d(d0, d0, kernel_size=2, stride=2)
        )
        self.final_conv = DoubleConv(d0, d0)
        self.head = OutConv(d0, num_classes)

        for module in (self.up3, self.up2, self.up1, self.up0, self.final_conv, self.head):
            module.apply(init_weights)

        if freeze_encoder:
            self.set_encoder_trainable(False)

        LOGGER.debug("Built UNetResNet34(pretrained=%s, bilinear=%s)", pretrained, bilinear)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient flow through the ResNet encoder."""
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for param in module.parameters():
                param.requires_grad = trainable
        LOGGER.info("Encoder parameters %s", "unfrozen" if trainable else "frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(N, C, H, W)`` inputs to ``(N, num_classes, H, W)`` logits."""
        if x.dim() != 4:
            raise ValueError(f"expected a 4-D (N, C, H, W) tensor, got {tuple(x.shape)}")
        input_size = x.shape[-2:]

        x0 = self.stem(x)        # /2
        x1 = self.layer1(x0)     # /4
        x2 = self.layer2(x1)     # /8
        x3 = self.layer3(x2)     # /16
        x4 = self.layer4(x3)     # /32

        feat = self.up3(x4, x3)
        feat = self.up2(feat, x2)
        feat = self.up1(feat, x1)
        feat = self.up0(feat, x0)

        feat = self.final_conv(self.final_up(feat))
        logits = self.head(feat)

        if logits.shape[-2:] != input_size:
            # Odd input sizes: the /32 encoder rounds down, so restore exactly.
            logits = nn.functional.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
