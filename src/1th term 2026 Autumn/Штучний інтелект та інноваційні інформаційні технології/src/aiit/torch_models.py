"""Архітектури нейронних мереж у PyTorch (завдання 6-7).

Архітектури навмисно повторюють tf_models.py шар у шар, щоб порівняння
фреймворків у лабораторній роботі №2 було коректним.
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
        raise ValueError(f"Невідома функція активації: {name}") from None


def build_baseline(input_dim: int, activation: str) -> nn.Module:
    """Базова feedforward мережа: два приховані шари, без регуляризації."""
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_UNITS[0]),
        _activation(activation),
        nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]),
        _activation(activation),
        nn.Linear(HIDDEN_UNITS[1], 1),
    )


def build_regularized(input_dim: int, activation: str) -> nn.Module:
    """Ускладнена мережа: Dropout + BatchNorm1d + L2 (через weight_decay)."""
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
    """Чотири конфігурації разом зі значенням weight_decay для кожної.

    У Keras L2 задається на рівні шару, у PyTorch — параметром оптимізатора
    weight_decay. Базові моделі регуляризації не мають, тому weight_decay=0.
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
