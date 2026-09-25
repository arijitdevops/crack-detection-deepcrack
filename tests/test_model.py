"""Architecture tests: output shapes, odd sizes, factory behaviour."""

from __future__ import annotations

import pytest
import torch

from src.model import available_models, build_model
from src.model.blocks import DoubleConv, Down, OutConv, Up, pad_to_match
from src.model.unet import UNet


@pytest.mark.parametrize("size", [(64, 64), (96, 128), (48, 48)])
def test_unet_preserves_spatial_size(size: tuple[int, int]) -> None:
    model = build_model("unet", base_channels=4, depth=2)
    model.eval()
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, *size)


@pytest.mark.parametrize("size", [(65, 65), (50, 71), (33, 97)])
def test_unet_handles_odd_input_sizes(size: tuple[int, int]) -> None:
    """Odd sizes lose pixels when pooling; the decoder must pad them back."""
    model = build_model("unet", base_channels=4, depth=3)
    model.eval()
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, *size)


def test_unet_transposed_conv_variant_matches_shape() -> None:
    model = UNet(base_channels=4, depth=2, bilinear=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 64, 80))
    assert y.shape == (2, 1, 64, 80)


def test_unet_depth_and_width_are_configurable() -> None:
    shallow = UNet(base_channels=4, depth=1)
    deep = UNet(base_channels=8, depth=3)
    shallow_params = sum(p.numel() for p in shallow.parameters())
    deep_params = sum(p.numel() for p in deep.parameters())
    assert deep_params > shallow_params


def test_unet_outputs_logits_not_probabilities() -> None:
    model = UNet(base_channels=4, depth=2)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 32, 32) * 5.0)
    # Logits are unconstrained; a probability map would never go negative.
    assert float(y.min()) < 0.0
    assert torch.isfinite(y).all()
    with torch.no_grad():
        probs = model.predict_proba(torch.randn(1, 3, 32, 32))
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_unet_rejects_non_4d_input() -> None:
    model = UNet(base_channels=4, depth=1)
    with pytest.raises(ValueError, match="4-D"):
        model(torch.randn(3, 32, 32))


def test_unet_resnet34_shape_without_pretrained_weights() -> None:
    model = build_model("unet_resnet34", pretrained=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, 1, 64, 64)


def test_unet_resnet34_handles_odd_sizes() -> None:
    model = build_model("unet_resnet34", pretrained=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 67, 93))
    assert y.shape == (1, 1, 67, 93)


def test_unet_resnet34_encoder_can_be_frozen() -> None:
    model = build_model("unet_resnet34", pretrained=False, freeze_encoder=True)
    assert all(not p.requires_grad for p in model.layer4.parameters())
    model.set_encoder_trainable(True)
    assert all(p.requires_grad for p in model.layer4.parameters())


def test_factory_lists_and_builds_every_registered_model() -> None:
    assert available_models() == ["unet", "unet_resnet34"]
    assert isinstance(build_model("UNet".lower(), base_channels=4, depth=1), UNet)


def test_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("segformer")


def test_factory_ignores_irrelevant_kwargs() -> None:
    # base_channels means nothing to the ResNet variant; it must not explode.
    model = build_model("unet_resnet34", pretrained=False, base_channels=99, depth=7)
    assert model.num_classes == 1


def test_blocks_produce_expected_channel_counts() -> None:
    x = torch.randn(1, 3, 32, 32)
    double = DoubleConv(3, 8)
    assert double(x).shape == (1, 8, 32, 32)

    down = Down(8, 16)
    assert down(double(x)).shape == (1, 16, 16, 16)

    up = Up(in_channels=16, skip_channels=8, out_channels=8, bilinear=True)
    assert up(down(double(x)), double(x)).shape == (1, 8, 32, 32)

    head = OutConv(8, 1)
    assert head(double(x)).shape == (1, 1, 32, 32)


def test_pad_to_match_pads_and_crops() -> None:
    reference = torch.zeros(1, 1, 10, 12)
    smaller = torch.zeros(1, 1, 9, 11)
    larger = torch.zeros(1, 1, 12, 14)
    assert pad_to_match(smaller, reference).shape == reference.shape
    assert pad_to_match(larger, reference).shape == reference.shape
    assert pad_to_match(reference, reference) is reference


# ---------------------------------------------------------------------- #
# Sliding-window inference
# ---------------------------------------------------------------------- #
class _ConstantModel(torch.nn.Module):
    """Emits a fixed logit everywhere, so blending can be checked exactly."""

    def __init__(self, value: float = 1.25) -> None:
        super().__init__()
        self.value = value
        self.probe = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)


def test_sliding_window_matches_direct_inference_on_a_small_image() -> None:
    """One tile covering the whole image must reproduce a direct forward pass."""
    from src.utils import sliding_window_inference

    torch.manual_seed(0)
    model = build_model("unet", base_channels=4, depth=2)
    model.eval()
    image = torch.randn(3, 64, 64)

    with torch.no_grad():
        direct = model(image.unsqueeze(0))
    tiled = sliding_window_inference(image, model, tile_size=64, overlap=0.0)

    assert tiled.shape == direct.shape
    assert torch.allclose(tiled, direct, atol=1e-5)


def test_sliding_window_blending_is_weight_normalised() -> None:
    """Overlapping tiles must average to the tile value, not accumulate."""
    from src.utils import sliding_window_inference

    model = _ConstantModel(value=1.25)
    output = sliding_window_inference(
        torch.randn(3, 160, 200), model, tile_size=64, overlap=0.5
    )
    assert output.shape == (1, 1, 160, 200)
    assert torch.allclose(output, torch.full_like(output, 1.25), atol=1e-4)


def test_sliding_window_pads_images_smaller_than_the_tile() -> None:
    from src.utils import sliding_window_inference

    model = _ConstantModel(value=-0.5)
    output = sliding_window_inference(torch.randn(3, 20, 31), model, tile_size=64, overlap=0.25)
    assert output.shape == (1, 1, 20, 31)


def test_sliding_window_rejects_invalid_overlap() -> None:
    from src.utils import sliding_window_inference

    with pytest.raises(ValueError, match="overlap"):
        sliding_window_inference(torch.randn(3, 32, 32), _ConstantModel(), overlap=1.0)
