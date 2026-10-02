"""Neural network architectures in TensorFlow/Keras (tasks 6-7).

TF_CPP_MIN_LOG_LEVEL is set by the aiit package (see __init__.py) — before any
`import tensorflow`, because here it would already be too late.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from aiit.config import (
    ACTIVATIONS,
    DROPOUT_RATE,
    HIDDEN_UNITS,
    L2_LAMBDA,
    LEARNING_RATE,
)


def build_baseline(input_dim: int, activation: str) -> keras.Model:
    """Baseline feedforward network: two hidden layers, no regularization."""
    return keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(HIDDEN_UNITS[0], activation=activation),
            layers.Dense(HIDDEN_UNITS[1], activation=activation),
            layers.Dense(1),
        ],
        name=f"baseline_{activation}",
    )


def build_regularized(input_dim: int, activation: str) -> keras.Model:
    """Extended network: Dropout + Batch Normalization + L2 regularization.

    BatchNormalization goes before the activation, which keeps the Dense layer
    linear and moves the activation out into a layer of its own.
    """
    reg = regularizers.l2(L2_LAMBDA)
    model = keras.Sequential(name=f"regularized_{activation}")
    model.add(keras.Input(shape=(input_dim,)))

    for units in HIDDEN_UNITS:
        model.add(layers.Dense(units, kernel_regularizer=reg, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(1, kernel_regularizer=reg))
    return model


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def all_models(input_dim: int) -> dict[str, keras.Model]:
    """Four configurations: 2 architectures × 2 activation functions."""
    models: dict[str, keras.Model] = {}
    for activation in ACTIVATIONS:
        models[f"baseline-{activation}"] = compile_model(
            build_baseline(input_dim, activation)
        )
        models[f"regularized-{activation}"] = compile_model(
            build_regularized(input_dim, activation)
        )
    return models


def set_seed(seed: int) -> None:
    """Seeds the TensorFlow generator. A separate function so that the call stays
    in the module that has already imported tensorflow."""
    tf.random.set_seed(seed)
