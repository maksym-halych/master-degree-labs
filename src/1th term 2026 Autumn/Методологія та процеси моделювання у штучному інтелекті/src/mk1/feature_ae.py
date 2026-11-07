"""The proposed model: an autoencoder that reconstructs patch embeddings.

The detector is trained on defect-free images only, as the benchmark demands.
A frozen backbone (mk1.patches) turns an image into a grid of patch
embeddings; a small convolutional autoencoder learns to squeeze every
embedding through a narrow bottleneck and rebuild it. It can only learn to
rebuild what it was shown, so a patch unlike every training patch — a defect —
comes out rebuilt badly, and its reconstruction error is its anomaly score
(feature reconstruction in the manner of DFR, Shi et al., 2021).

Reconstructing features rather than pixels is the point of the design. A
pixel autoencoder blurs every fine detail, so it also "detects" the normal
texture it could not rebuild; a feature autoencoder works where the backbone
has already discarded lighting and noise and kept what describes the content.

The bottleneck is per patch (1×1 convolutions at the ends) with a 3×3
convolution on either side of it, so a patch is rebuilt from its neighbourhood
too: a defect cannot be reproduced by copying the patch itself through.
Two regularisers guard against the autoencoder generalising so well that it
rebuilds defects as well: weight decay, and masking noise that zeroes a share
of the input features during training, which the autoencoder has to restore
from the rest (a denoising autoencoder, Vincent et al., 2008).
"""

import copy
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from torch import nn
from torch.nn import functional as F

from mk1.anomaly_metrics import pixel_auroc
from mk1.config import SEED
from mk1.mvtec import Part
from mk1.patches import PatchEmbedder, aggregate, select

log = logging.getLogger(__name__)

HIDDEN = 256
LATENT = 64

EPOCHS = 30
BATCH = 8  # images per step, that is 8 × 32 × 32 patch embeddings
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class Config:
    """
    One point of the hyperparameter study.

    Args:
        backbone: timm name of the feature extractor, a key of BACKBONES.
        patch_size: Neighbourhood, in feature-map cells, each embedding averages.
        weight_decay: AdamW's decoupled weight decay.
        dropout: Share of input features zeroed during training (masking noise).
    """

    backbone: str
    patch_size: int = 3
    weight_decay: float = 1e-2
    dropout: float = 0.0


def conv_block(source: int, target: int, kernel: int) -> nn.Sequential:
    """Convolution, batch normalisation and ReLU, keeping the grid size."""
    return nn.Sequential(
        nn.Conv2d(source, target, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(target),
        nn.ReLU(inplace=True),
    )


class FeatureAutoencoder(nn.Module):
    """
    Rebuilds standardised patch embeddings through a per-patch bottleneck.

    Args:
        channels: Size of one patch embedding.
        dropout: Share of input features zeroed during training.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        # Per-channel statistics of the training embeddings. Channels differ in
        # scale by orders of magnitude, and unstandardised the loss would be
        # dominated by the few largest.
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("std", torch.ones(1, channels, 1, 1))
        self.encoder = nn.Sequential(
            conv_block(channels, HIDDEN, 1),
            conv_block(HIDDEN, HIDDEN // 2, 3),
            nn.Conv2d(HIDDEN // 2, LATENT, 1),
        )
        self.decoder = nn.Sequential(
            conv_block(LATENT, HIDDEN // 2, 1),
            conv_block(HIDDEN // 2, HIDDEN, 3),
            nn.Conv2d(HIDDEN, channels, 1),
        )

    def standardise(self, embedding: torch.Tensor) -> torch.Tensor:
        """Embeddings in units of the training set's per-channel spread."""
        return (embedding - self.mean) / self.std

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct standardised embeddings.

        Args:
            target: Standardised embeddings, (N, channels, H, W).

        Returns:
            The reconstruction, same shape.
        """
        source = target
        if self.training and self.dropout > 0:
            # Masking noise without rescaling: the zeroed features are the
            # corruption to undo, not a stochastic regulariser to average out.
            source = target * (torch.rand_like(target) >= self.dropout)
        return self.decoder(self.encoder(source))

    def patch_scores(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction error of each patch, the anomaly score before upsampling.

        Args:
            embedding: Raw patch embeddings, (N, channels, H, W).

        Returns:
            Mean squared error over the channels, (N, 1, H, W).
        """
        target = self.standardise(embedding)
        return (self(target) - target).pow(2).mean(dim=1, keepdim=True)


class FeatureAEDetector(nn.Module):
    """
    The whole inference path: image in, anomaly map out.

    Upsampling and smoothing are PatchCore's own (nearest-neighbour upsampling,
    Gaussian blur with σ = 4), so the maps of the two models differ only in
    how a patch is scored.

    Args:
        embedder: The frozen backbone.
        autoencoder: The trained autoencoder.
    """

    def __init__(
        self, embedder: PatchEmbedder, autoencoder: FeatureAutoencoder
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.autoencoder = autoencoder
        self.map_generator = AnomalyMapGenerator()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Images (N, 3, H, W) to anomaly maps (N, 1, H, W)."""
        with torch.no_grad():
            scores = self.autoencoder.patch_scores(self.embedder(images))
        return self.map_generator(scores, images.shape[-2:])


def fit_standardisation(
    autoencoder: FeatureAutoencoder, features: list[torch.Tensor], patch_size: int
) -> None:
    """
    Set the autoencoder's per-channel mean and std from the training embeddings.

    Args:
        autoencoder: The model to update in place.
        features: Raw layer features of the training images.
        patch_size: Neighbourhood the embeddings are averaged over.
    """
    total = torch.zeros(autoencoder.mean.shape[1], dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for start in range(0, len(features[0]), BATCH):
        embedding = aggregate(select(features, slice(start, start + BATCH)), patch_size)
        flat = embedding.transpose(0, 1).flatten(1).double()
        total += flat.sum(dim=1)
        squares += flat.pow(2).sum(dim=1)
        count += flat.shape[1]
    mean = total / count
    std = (squares / count - mean.pow(2)).clamp_min(0).sqrt()
    # A channel the ReLU switched off on every training patch has no spread;
    # flooring it keeps the standardisation finite.
    autoencoder.mean.copy_(mean.float().view_as(autoencoder.mean))
    autoencoder.std.copy_(std.clamp_min(1e-6).float().view_as(autoencoder.std))


def anomaly_maps(
    autoencoder: FeatureAutoencoder,
    features: list[torch.Tensor],
    patch_size: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    Anomaly maps of a whole split from its cached layer features.

    Args:
        autoencoder: The model, switched to evaluation mode here.
        features: Raw layer features of the split's images.
        patch_size: Neighbourhood the embeddings are averaged over.
        image_size: Height and width of the maps.

    Returns:
        Maps of shape (N, 1, *image_size).
    """
    autoencoder.eval()
    generator = AnomalyMapGenerator()
    maps = []
    with torch.no_grad():
        for start in range(0, len(features[0]), BATCH):
            embedding = aggregate(
                select(features, slice(start, start + BATCH)), patch_size
            )
            maps.append(generator(autoencoder.patch_scores(embedding), image_size))
    return torch.cat(maps)


def reconstruction_loss(
    autoencoder: FeatureAutoencoder, features: list[torch.Tensor], patch_size: int
) -> float:
    """Mean reconstruction error, in evaluation mode, over every patch of a split."""
    autoencoder.eval()
    losses, weights = [], []
    with torch.no_grad():
        for start in range(0, len(features[0]), BATCH):
            embedding = aggregate(
                select(features, slice(start, start + BATCH)), patch_size
            )
            target = autoencoder.standardise(embedding)
            losses.append(F.mse_loss(autoencoder(target), target).item())
            weights.append(len(embedding))
    return float(np.average(losses, weights=weights))


def fit(
    config: Config,
    channels: int,
    train_features: list[torch.Tensor],
    val_features: list[torch.Tensor],
    val: Part,
    epochs: int = EPOCHS,
) -> tuple[FeatureAutoencoder, list[dict]]:
    """
    Train an autoencoder on the defect-free training embeddings.

    After every epoch the model is scored on the validation part: the
    reconstruction loss on its defect-free images, which shows whether the
    autoencoder generalises to normal images it was not trained on, and the
    Pixel_AUROC on all of it, which shows whether it tells defects apart. The
    returned model is the one of the epoch with the best validation
    Pixel_AUROC — early stopping in hindsight, chosen on validation only.

    Args:
        config: The hyperparameters.
        channels: Size of one patch embedding.
        train_features: Raw layer features of the training images.
        val_features: Raw layer features of the validation images.
        val: The validation part, for its masks and defect types.
        epochs: Number of passes over the training images.

    Returns:
        The best model and one record per epoch.
    """
    torch.manual_seed(SEED)
    autoencoder = FeatureAutoencoder(channels, dropout=config.dropout)
    fit_standardisation(autoencoder, train_features, config.patch_size)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=config.weight_decay
    )
    image_size = tuple(val.masks.shape[-2:])
    good = torch.from_numpy(~val.anomalous)
    val_good = select(val_features, good)
    images = len(train_features[0])

    history: list[dict] = []
    best_state, best_auroc = None, -1.0
    started = time.time()
    for epoch in range(1, epochs + 1):
        autoencoder.train()
        order = torch.randperm(images)
        losses = []
        for start in range(0, images, BATCH):
            batch = select(train_features, order[start : start + BATCH])
            target = autoencoder.standardise(aggregate(batch, config.patch_size))
            loss = F.mse_loss(autoencoder(target), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        maps = anomaly_maps(autoencoder, val_features, config.patch_size, image_size)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": reconstruction_loss(autoencoder, val_good, config.patch_size),
            "val_pixel_auroc": pixel_auroc(maps, val),
            "seconds": time.time() - started,
        }
        history.append(record)
        log.debug("%s epoch %d: %s", asdict(config), epoch, record)
        if record["val_pixel_auroc"] > best_auroc:
            best_auroc = record["val_pixel_auroc"]
            best_state = copy.deepcopy(autoencoder.state_dict())

    autoencoder.load_state_dict(best_state)
    autoencoder.eval()
    return autoencoder, history
