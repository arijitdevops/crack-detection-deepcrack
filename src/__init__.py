"""Crack segmentation on the DeepCrack dataset.

Public subpackages:
    :mod:`src.model`   -- U-Net architectures and the ``build_model`` factory.

Public modules:
    :mod:`src.config`     -- YAML + environment configuration.
    :mod:`src.dataset`    -- :class:`~src.dataset.CrackSegmentationDataset`.
    :mod:`src.transforms` -- joint image/mask augmentation.
    :mod:`src.losses`     -- Dice / BCE-Dice / Focal / Tversky objectives.
    :mod:`src.metrics`    -- confusion-matrix based segmentation metrics.
    :mod:`src.engine`     -- train/validate loops.
    :mod:`src.utils`      -- seeding, checkpoints, overlays, tiled inference.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
