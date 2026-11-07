"""Computational cost of a detector: parameters, FLOPs and CPU latency per image.

Every detector is measured as a whole, from a normalised image to its anomaly
map, because that is what a deployment pays for: for PatchCore most of the cost
is the nearest-neighbour search after the backbone, which a count of the
backbone alone would hide.

FLOPs are counted by torch's FlopCounterMode, which records every matrix
multiplication and convolution actually dispatched, so the nearest-neighbour
search and the Mahalanobis distance are counted along with the networks. One
multiply-add is two FLOPs. Element-wise operations are not counted; they are a
small share of every model here.

Latency is wall-clock time on the CPU, at batch size 1 — one image at a time,
as an inspection line delivers them — with the thread count pinned so that the
numbers of different models are comparable. The machine has no GPU usable by
torch, so only CPU figures are reported. Nothing else heavy should run beside
the measurement.
"""

import logging
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

log = logging.getLogger(__name__)

# One per physical core: the second hardware thread of a core shares its
# vector units, and torch's default on this machine is the same number.
THREADS = 8

WARMUP = 5
REPEATS = 30


def cpu_name() -> str:
    """The processor's model name, for the report's hardware line."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def count_parameters(model: nn.Module, backbone: nn.Module) -> dict[str, int]:
    """
    Parameters and buffers of a detector, the frozen backbone apart.

    The backbone is told apart by module, not by requires_grad: Anomalib's
    extractor freezes it by running under no_grad and leaves the flag set.
    Buffers count too: PaDiM's Gaussians and PatchCore's memory bank are
    buffers, not parameters, yet they are the model as much as the weights are.

    Args:
        model: The detector.
        backbone: Its pretrained feature extractor.

    Returns:
        Backbone and head parameter counts and the number of buffer elements.
    """
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in backbone.parameters())
    buffers = sum(b.numel() for b in model.buffers()) - sum(
        b.numel() for b in backbone.buffers()
    )
    return {"backbone": frozen, "head": total - frozen, "buffers": buffers}


def count_flops(model: nn.Module, image: torch.Tensor) -> int:
    """
    FLOPs of one forward pass.

    Args:
        model: The detector, in evaluation mode.
        image: A batch of one normalised image.

    Returns:
        The number of floating-point operations.
    """
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(image)
    return counter.get_total_flops()


def measure_latency(model: nn.Module, image: torch.Tensor) -> dict[str, float]:
    """
    Wall-clock latency of one forward pass at batch size 1.

    Args:
        model: The detector, in evaluation mode.
        image: A batch of one normalised image.

    Returns:
        Median, 10th and 90th percentile, in milliseconds per image.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        timings = []
        with torch.no_grad():
            for run in range(WARMUP + REPEATS):
                started = time.perf_counter()
                model(image)
                elapsed = (time.perf_counter() - started) * 1000
                if run >= WARMUP:
                    timings.append(elapsed)
    finally:
        torch.set_num_threads(previous)
    deciles = statistics.quantiles(timings, n=10)
    return {
        "median_ms": statistics.median(timings),
        "p10_ms": deciles[0],
        "p90_ms": deciles[-1],
    }


def profile(
    name: str, model: nn.Module, backbone: nn.Module, image: torch.Tensor
) -> dict:
    """
    Every cost figure of a detector.

    The backbone is profiled on its own as well, so the share of the cost that
    comes after it — the autoencoder, the Gaussians, the neighbour search — can
    be read off as the difference.

    Args:
        name: For the log.
        model: The detector, in evaluation mode.
        backbone: Its pretrained feature extractor.
        image: A batch of one normalised image.

    Returns:
        Parameters, GFLOPs per image and latency, of the detector and its backbone.
    """
    model.eval()
    result = {
        "parameters": count_parameters(model, backbone),
        "gflops": count_flops(model, image) / 1e9,
        "latency": measure_latency(model, image),
        "backbone_gflops": count_flops(backbone, image) / 1e9,
        "backbone_latency": measure_latency(backbone, image),
    }
    log.info(
        "%-28s %6.2f GFLOPs  %7.1f ms/image (p10 %.1f, p90 %.1f); backbone %.2f GFLOPs, %.1f ms",
        name,
        result["gflops"],
        result["latency"]["median_ms"],
        result["latency"]["p10_ms"],
        result["latency"]["p90_ms"],
        result["backbone_gflops"],
        result["backbone_latency"]["median_ms"],
    )
    return result
