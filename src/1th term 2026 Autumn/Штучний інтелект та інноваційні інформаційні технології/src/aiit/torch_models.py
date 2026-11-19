"""Neural network architectures in PyTorch (tasks 6-7).

The architectures deliberately mirror tf_models.py layer for layer, so that the
framework comparison in lab 2 is a fair one.
"""

import torch
import torch.nn as nn

from aiit.config import ACTIVATIONS, DROPOUT_RATE, HIDDEN_UNITS, L2_LAMBDA, SEED

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"Unknown activation function: {name}") from None


def build_baseline(input_dim: int, activation: str) -> nn.Module:
    """Baseline feedforward network: two hidden layers, no regularization."""
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS[0]),
        _activation(activation),
        nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]),
        _activation(activation),
        nn.Linear(HIDDEN_UNITS[1], 1),
    )


def build_regularized(input_dim: int, activation: str) -> nn.Module:
    """Extended network: Dropout + BatchNorm1d + L2 (through weight_decay)."""
    layers: list[nn.Module] = []
    prev = input_dim

    for units in HIDDEN_UNITS:
        layers += [
            nn.Linear(prev, units, bias=False),
            nn.BatchNorm1d(units),
            _activation(activation),
            nn.Dropout(DROPOUT_RATE),
        ]
        prev = units

    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def all_models(input_dim: int) -> dict[str, tuple[nn.Module, float]]:
    """Four configurations together with the weight_decay value for each.

    In Keras L2 is declared per layer, in PyTorch it is the optimizer's
    weight_decay parameter. The baseline models carry no regularization, hence
    weight_decay=0.
    """
    models: dict[str, tuple[nn.Module, float]] = {}
    for activation in ACTIVATIONS:
        torch.manual_seed(SEED)
        models[f"baseline-{activation}"] = (build_baseline(input_dim, activation), 0.0)
        torch.manual_seed(SEED)
        models[f"regularized-{activation}"] = (
            build_regularized(input_dim, activation),
            L2_LAMBDA,
        )
    return models


def build_mlp(
    input_dim: int,
    hidden: tuple[int, ...],
    activation: str,
    dropout: float,
    batch_norm: bool,
) -> nn.Module:
    """
    Build a feedforward network of any depth from a set of hyperparameters (lab 3).

    The lab 2 architectures are points of this family: baseline-relu is
    hidden=(64, 32) with ReLU, no Dropout and no BatchNorm, and it is built
    layer for layer the same, so the same seed gives it the same weights.

    Args:
        input_dim: Number of input features.
        hidden: Width of every hidden layer, input side first.
        activation: Name of the activation function, a key of _ACTIVATIONS.
        dropout: Dropout rate after every hidden layer; 0 leaves Dropout out.
        batch_norm: Whether to put BatchNorm1d before every activation.

    Returns:
        The untrained network.

    Raises:
        ValueError: If the activation function is unknown.
    """
    layers: list[nn.Module] = []
    prev = input_dim

    for units in hidden:
        # The bias is redundant before BatchNorm: its own shift beta replaces it.
        layers.append(nn.Linear(prev, units, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm1d(units))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = units

    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)
