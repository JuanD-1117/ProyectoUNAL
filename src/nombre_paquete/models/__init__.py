"""
Definición del modelo baseline (CNN sencilla) para clasificación de galaxias.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    """
    Construye el modelo baseline CNN descrito en la documentación.

    Parameters
    ----------
    input_shape : tuple
        Forma de entrada de las imágenes (alto, ancho, canales), por ejemplo (128, 128, 3).
    num_classes : int
        Número de clases de salida (en este proyecto, 5).

    Returns
    -------
    tf.keras.Model
        Modelo Keras compilable/entrenable.
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
