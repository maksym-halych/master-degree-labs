"""Patch embeddings from frozen ImageNet backbones: the feature extraction of lab 4.

A backbone's feature map at stride 8 cuts a 256-pixel image into a 32×32 grid of
patches, one feature vector per 8×8-pixel cell. Two layers are taken, as in
PatchCore (Roth et al., 2022): the one at stride 8 keeps local texture, the one
at stride 16 adds context, and it is upsampled onto the first one's grid and
concatenated. The patch size p averages each layer over a p×p neighbourhood of
cells before the concatenation, so a vector describes a region of p·8 pixels
plus the receptive field: a larger p tolerates small misalignments and noise, a
smaller one keeps a small defect from being averaged away.

The backbones are never trained. Their weights are ImageNet's, so a defect is
not something they were taught to see: it is a patch whose features fall where
the defect-free training patches never did.
"""

import logging
from dataclasses import dataclass

import torch
from anomalib.models.components import TimmFeatureExtractor
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)

# Images per forward pass of a backbone: large enough to keep the CPU cores
# busy, small enough that WideResNet-50's activations fit beside the rest.
EXTRACT_BATCH = 16


@dataclass(frozen=True)
class Backbone:
    """
    A pretrained network and the two layers taken from it.

    Args:
        name: The timm model name.
        layers: The stride-8 and stride-16 feature layers, in that order.
        label: How figures and tables name it.
    """

    name: str
    layers: tuple[str, str]
    label: str


BACKBONES = {
    backbone.name: backbone
    for backbone in (
        Backbone("resnet18", ("layer2", "layer3"), "ResNet-18"),
        Backbone("wide_resnet50_2", ("layer2", "layer3"), "WideResNet-50-2"),
        # MobileNetV3's stride-8 and stride-16 outputs end blocks 2 and 4.
        Backbone(
            "mobilenetv3_large_100", ("blocks.2", "blocks.4"), "MobileNetV3-Large"
        ),
    )
}


class PatchEmbedder(nn.Module):
    """
    A frozen backbone that turns images into a grid of patch embeddings.

    Args:
        backbone: Which network and layers to use.
        patch_size: Side of the neighbourhood, in feature-map cells, that each
            layer is averaged over; odd, so the neighbourhood has a centre.

    Raises:
        ValueError: If patch_size is not a positive odd number.
    """

    def __init__(self, backbone: Backbone, patch_size: int) -> None:
        super().__init__()
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError(
                f"patch_size must be a positive odd number, got {patch_size}"
            )
        self.backbone = backbone
        self.patch_size = patch_size
        self.extractor = TimmFeatureExtractor(
            backbone=backbone.name, layers=list(backbone.layers), pre_trained=True
        ).eval()
        self.channels = sum(self.extractor.out_dims)

    def layer_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """The raw outputs of the two layers, stride 8 first."""
        with torch.no_grad():
            features = self.extractor(images)
        return [features[layer] for layer in self.backbone.layers]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Images (N, 3, H, W) to patch embeddings (N, channels, H/8, W/8)."""
        return aggregate(self.layer_features(images), self.patch_size)


def aggregate(features: list[torch.Tensor], patch_size: int) -> torch.Tensor:
    """
    Pool each layer over the patch neighbourhood and stack them on one grid.

    A function rather than a method so that one backbone pass serves every
    patch size of the hyperparameter study.

    Args:
        features: Outputs of PatchEmbedder.layer_features(), stride 8 first.
        patch_size: Side of the neighbourhood, in feature-map cells.

    Returns:
        Embeddings of shape (N, channels, H, W), on the stride-8 grid.
    """
    size = features[0].shape[-2:]
    pooled = []
    for feature in features:
        if patch_size > 1:
            # count_include_pad=False: a border cell averages only the cells
            # that exist instead of being diluted by zero padding.
            feature = F.avg_pool2d(
                feature,
                kernel_size=patch_size,
                stride=1,
                padding=patch_size // 2,
                count_include_pad=False,
            )
        if feature.shape[-2:] != size:
            feature = F.interpolate(
                feature, size=size, mode="bilinear", align_corners=False
            )
        pooled.append(feature)
    return torch.cat(pooled, dim=1)


def extract(embedder: PatchEmbedder, images: torch.Tensor) -> list[torch.Tensor]:
    """
    Run the backbone once over a whole split and keep the raw layer outputs.

    The patch size is applied afterwards by aggregate(), batch by batch, so one
    extraction serves every patch size and every training epoch: the backbone
    is by far the most expensive part, and it never changes.

    Args:
        embedder: The frozen backbone.
        images: Normalised images of one split.

    Returns:
        One tensor per layer, holding that layer's output for every image.
    """
    outputs: list[list[torch.Tensor]] = [[] for _ in embedder.backbone.layers]
    for start in range(0, len(images), EXTRACT_BATCH):
        batch = embedder.layer_features(images[start : start + EXTRACT_BATCH])
        for store, feature in zip(outputs, batch):
            store.append(feature)
    features = [torch.cat(store) for store in outputs]
    log.debug(
        "%s features of %d images: %s",
        embedder.backbone.name,
        len(images),
        [tuple(f.shape[1:]) for f in features],
    )
    return features


def select(
    features: list[torch.Tensor], index: torch.Tensor | slice
) -> list[torch.Tensor]:
    """The same images taken from every layer of an extract() result."""
    return [feature[index] for feature in features]
