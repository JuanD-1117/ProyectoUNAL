"""
Definición del modelo final basado en EfficientNetB0 con transfer learning.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    train_base: bool = False,
) -> tf.keras.Model:
    """
    Construye un modelo de clasificación basado en EfficientNetB0.

    Parameters
    ----------
    input_shape : tuple
        Forma de entrada de las imágenes, por ejemplo (128, 128, 3).
    num_classes : int
        Número de clases de salida.
    train_base : bool, optional
        Si True, la base convolucional se entrena (fine-tuning). Si False,
        la base se congela y solo se entrena la cabeza de clasificación.

    Returns
    -------
    tf.keras.Model
        Modelo Keras listo para compilar y entrenar.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # use base in inference mode para estabilidad
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
