"""
Funciones de evaluación para modelos de clasificación de galaxias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from nombre_paquete.training.data_pipeline import (
    IMG_SIZE,
    create_datasets,
    get_project_root,
)


def load_model(model_path: Path) -> tf.keras.Model:
    """
    Carga un modelo Keras desde un archivo.

    Parameters
    ----------
    model_path : Path
        Ruta al archivo del modelo (.keras).

    Returns
    -------
    tf.keras.Model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    return tf.keras.models.load_model(model_path)


def evaluate_model(
    model: tf.keras.Model,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evalúa un modelo sobre el conjunto de validación y devuelve y_true / y_pred.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo ya cargado.
    batch_size : int, optional
        Tamaño de lote a usar en el dataset.
    val_split : float, optional
        Proporción de datos para validación (debe coincidir con el entrenamiento).

    Returns
    -------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Etiquetas predichas.
    """
    # Recrear datasets (mismo split)
    _, val_ds, class_to_index, index_to_class = create_datasets(
        batch_size=batch_size,
        val_split=val_split,
    )

    # Recopilar predicciones
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds_classes = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds_classes.tolist())

    return np.array(y_true), np.array(y_pred), index_to_class


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_class: Dict[int, str],
) -> str:
    """
    Genera un classification_report legible usando nombres de clases.

    Returns
    -------
    str
        Texto del classification_report.
    """
    target_names = [index_to_class[i] for i in sorted(index_to_class.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calcula la matriz de confusión.

    Returns
    -------
    np.ndarray
        Matriz de confusión (n_clases x n_clases).
    """
    return confusion_matrix(y_true, y_pred)
