"""
Módulo de construcción del pipeline de datos para entrenamiento y validación.

Responsabilidades:
- Cargar el dataset filtrado `merged_filtered.csv`.
- Crear el mapeo entre clases (gz2_class) e índices numéricos.
- Dividir el dataset en entrenamiento y validación.
- Construir objetos tf.data.Dataset listos para usar en el entrenamiento
  de modelos de clasificación de imágenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf

# Tamaño de imagen utilizado en el proyecto
IMG_SIZE: Tuple[int, int] = (128, 128)


def get_project_root() -> Path:
    """
    Devuelve la ruta a la raíz del proyecto.

    Asume que este archivo se encuentra en:
        <root>/src/nombre_paquete/training/data_pipeline.py
    """
    return Path(__file__).resolve().parents[3]


def load_filtered_dataframe() -> pd.DataFrame:
    """
    Carga el archivo CSV filtrado generado en la etapa de adquisición de datos.

    Returns
    -------
    pandas.DataFrame
        DataFrame con al menos las columnas:
        - objid
        - gz2_class
        - image_path
    """
    project_root = get_project_root()
    csv_path = project_root / "data" / "interim" / "merged_filtered.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset filtrado en {csv_path}. "
            "Asegúrate de ejecutar primero scripts/data_acquisition/main.py."
        )

    df = pd.read_csv(csv_path)
    return df


def build_class_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construye los diccionarios de mapeo entre nombre de clase y índice numérico.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna `gz2_class`.

    Returns
    -------
    class_to_index : dict
        Mapeo de nombre de clase a índice numérico.
    index_to_class : dict
        Mapeo inverso de índice numérico a nombre de clase.
    """
    class_names = sorted(df["gz2_class"].unique())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}
    return class_to_index, index_to_class


def _load_and_preprocess_image(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Función utilizada por tf.data para cargar y preprocesar una imagen.

    Parameters
    ----------
    image_path : tf.Tensor
        Ruta a un archivo de imagen (tensor de tipo string).
    label : tf.Tensor
        Etiqueta numérica correspondiente.

    Returns
    -------
    image : tf.Tensor
        Imagen preprocesada, tamaño IMG_SIZE y valores en [0, 1].
    label : tf.Tensor
        Etiqueta numérica sin modificar.
    """
    # Leer contenido del archivo
    img = tf.io.read_file(image_path)
    # Decodificar JPEG a tensor RGB
    img = tf.image.decode_jpeg(img, channels=3)
    # Redimensionar
    img = tf.image.resize(img, IMG_SIZE)
    # Normalizar
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _make_dataset(
    image_paths,
    labels,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset a partir de listas de rutas y etiquetas.

    Parameters
    ----------
    image_paths : Sequence[str]
        Rutas a las imágenes.
    labels : Sequence[int]
        Etiquetas numéricas correspondientes.
    batch_size : int
        Tamaño de lote para el entrenamiento.
    training : bool
        Si True, se aplica barajado (shuffle).

    Returns
    -------
    tf.data.Dataset
        Dataset listo para usar en model.fit().
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    ds = ds.map(
        _load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_datasets(
    batch_size: int = 32,
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int], Dict[int, str]]:
    """
    Crea los datasets de entrenamiento y validación a partir de merged_filtered.csv.

    Parameters
    ----------
    batch_size : int, optional
        Tamaño de lote.
    val_split : float, optional
        Proporción de datos reservada para validación (entre 0 y 1).
    random_state : int, optional
        Semilla para el barajado de filas.

    Returns
    -------
    train_ds : tf.data.Dataset
        Dataset de entrenamiento.
    val_ds : tf.data.Dataset
        Dataset de validación.
    class_to_index : dict
        Mapeo de nombre de clase a índice numérico.
    index_to_class : dict
        Mapeo inverso de índice numérico a nombre de clase.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split debe estar entre 0 y 1.")

    df = load_filtered_dataframe()

    # Construir mapeos de clases
    class_to_index, index_to_class = build_class_mappings(df)

    # Añadir columna de etiquetas numéricas
    df = df.copy()
    df["label_idx"] = df["gz2_class"].map(class_to_index)

    # Barajar filas para split estratificado simple
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_total = len(df)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["label_idx"].tolist()

    val_paths = val_df["image_path"].tolist()
    val_labels = val_df["label_idx"].tolist()

    train_ds = _make_dataset(train_paths, train_labels, batch_size, training=True)
    val_ds = _make_dataset(val_paths, val_labels, batch_size, training=False)

    return train_ds, val_ds, class_to_index, index_to_class
