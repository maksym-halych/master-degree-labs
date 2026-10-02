"""Архітектури нейронних мереж у TensorFlow/Keras (завдання 6-7).

TF_CPP_MIN_LOG_LEVEL виставляє пакет aiit (див. __init__.py) — до будь-якого
`import tensorflow`, бо тут було б уже запізно.
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
    """Базова feedforward мережа: два приховані шари, без регуляризації."""
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
    """Ускладнена мережа: Dropout + Batch Normalization + L2-регуляризація.

    BatchNormalization ставимо перед активацією, тому Dense-шар лишається
    лінійним, а активація виноситься окремим шаром.
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
    """Чотири конфігурації: 2 архітектури × 2 функції активації."""
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
    """Фіксує генератор TensorFlow. Окрема функція, щоб виклик лишався
    в модулі, який уже імпортував tensorflow."""
    tf.random.set_seed(seed)
