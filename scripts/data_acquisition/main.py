"""
Script de adquisición y preparación inicial de datos para el proyecto Galaxy Zoo 2.

Responsabilidades principales:
1. Descargar el dataset de imágenes desde Kaggle (si no está presente).
2. Descargar el archivo gz2_hart16.csv.gz con las etiquetas de Hart 2016 (si no está presente).
3. Cargar el archivo de mapeo de nombres de archivo e IDs de galaxia.
4. Unir el mapeo con las etiquetas de Hart utilizando el identificador de objeto.
5. Construir la columna `image_path` con la ruta completa del archivo de imagen.
6. Filtrar las 5 clases más frecuentes de `gz2_class`.
7. Guardar el DataFrame resultante en `data/interim/merged_filtered.csv`.

Este script está pensado para ejecutarse desde la raíz del proyecto:

    python scripts/data_acquisition/main.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import kagglehub
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configura el sistema de logging del módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Funciones auxiliares de paths
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Devuelve la ruta a la raíz del proyecto.

    Se asume que este archivo está en:
        <root>/scripts/data_acquisition/main.py
    """
    return Path(__file__).resolve().parents[2]


def get_data_dirs(project_root: Path) -> Tuple[Path, Path]:
    """
    Devuelve las rutas de las carpetas de datos `raw` e `interim`.

    Si no existen, las crea.
    """
    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, interim_dir


# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------

def download_hart_labels(hart_url: str, dest_path: Path) -> Path:
    """
    Descarga el archivo gz2_hart16.csv.gz si no existe en disco.

    Parameters
    ----------
    hart_url : str
        URL desde la cual se descargará el archivo .csv.gz.
    dest_path : Path
        Ruta local donde se guardará el archivo.

    Returns
    -------
    Path
        Ruta local al archivo .csv.gz.
    """
    if dest_path.exists():
        LOGGER.info("Archivo Hart 2016 ya existe en %s, no se descarga de nuevo.", dest_path)
        return dest_path

    LOGGER.info("Descargando archivo Hart 2016 desde %s ...", hart_url)
    response = requests.get(hart_url, timeout=60)
    response.raise_for_status()

    dest_path.write_bytes(response.content)
    LOGGER.info("Archivo Hart 2016 guardado en %s", dest_path)

    return dest_path


def download_kaggle_dataset(dataset_name: str) -> Path:
    """
    Descarga el dataset de Kaggle usando kagglehub, si no está ya descargado.

    Parameters
    ----------
    dataset_name : str
        Nombre del dataset en Kaggle, por ejemplo:
        "jaimetrickz/galaxy-zoo-2-images"

    Returns
    -------
    Path
        Ruta local al directorio del dataset.
    """
    LOGGER.info("Descargando dataset de Kaggle: %s ...", dataset_name)
    dataset_path_str = kagglehub.dataset_download(dataset_name)
    dataset_path = Path(dataset_path_str)
    LOGGER.info("Dataset descargado en: %s", dataset_path)
    return dataset_path


# ---------------------------------------------------------------------------
# Carga y transformación de datos
# ---------------------------------------------------------------------------

def load_filename_mapping(mapping_path: Path) -> pd.DataFrame:
    """
    Carga el archivo de mapeo de nombres de archivo e IDs de galaxia.

    Se asume un CSV similar a `gz2_filename_mapping.csv`.

    Parameters
    ----------
    mapping_path : Path
        Ruta local al archivo de mapeo.

    Returns
    -------
    pandas.DataFrame
        DataFrame con la información de mapeo.
    """
    LOGGER.info("Cargando archivo de mapeo desde %s ...", mapping_path)
    mapping_df = pd.read_csv(mapping_path)
    LOGGER.info("Archivo de mapeo cargado con forma %s", mapping_df.shape)
    return mapping_df


def load_hart_dataframe(hart_gz_path: Path) -> pd.DataFrame:
    """
    Carga el archivo comprimido con las etiquetas de Hart 2016.

    Parameters
    ----------
    hart_gz_path : Path
        Ruta local al archivo .csv.gz.

    Returns
    -------
    pandas.DataFrame
        DataFrame con las etiquetas morfológicas.
    """
    LOGGER.info("Cargando etiquetas de Hart 2016 desde %s ...", hart_gz_path)
    hart_df = pd.read_csv(hart_gz_path, compression="gzip")
    LOGGER.info("Etiquetas Hart cargadas con forma %s", hart_df.shape)
    return hart_df


def merge_mapping_and_labels(mapping_df: pd.DataFrame, hart_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une el archivo de mapeo con las etiquetas de Hart 2016.

    NOTA IMPORTANTE:
    - En el notebook original se utiliza una unión entre el identificador de
      la galaxia en el mapeo y la columna `dr7objid` del archivo de Hart.
    - Aquí renombramos `dr7objid` a `objid` para poder unirlo con el mapeo.
    - Asegúrate de que los nombres de columnas coinciden con los del dataset
      real; si no, ajusta estos nombres.

    Returns
    -------
    pandas.DataFrame
        DataFrame resultante de la unión (merge).
    """
    hart_df = hart_df.rename(columns={"dr7objid": "objid"})

    LOGGER.info("Realizando merge entre mapping y etiquetas Hart ...")
    merged = pd.merge(mapping_df, hart_df, on="objid", how="inner")
    LOGGER.info("Merge completado. Forma resultante: %s", merged.shape)
    return merged


def add_image_paths(merged_df: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    """
    Añade la columna `image_path` al DataFrame mergeado.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame resultante de la unión mapping + Hart.
        Se asume que contiene la columna `asset_id`.
    images_root : Path
        Directorio donde se almacenan las imágenes .jpg.

    Returns
    -------
    pandas.DataFrame
        DataFrame con la columna adicional `image_path`.
    """
    LOGGER.info("Añadiendo columna 'image_path' usando images_root=%s ...", images_root)

    def build_image_path(asset_id: str) -> str:
        return str(images_root / f"{asset_id}.jpg")

    merged_df = merged_df.copy()
    merged_df["image_path"] = merged_df["asset_id"].apply(build_image_path)

    LOGGER.info("Columna 'image_path' añadida.")
    return merged_df


def filter_top_classes(
    merged_df: pd.DataFrame,
    target_col: str = "gz2_class",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Filtra el DataFrame para conservar únicamente las `top_n` clases más
    frecuentes de la columna objetivo.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame con la columna de clases.
    target_col : str, optional
        Nombre de la columna que contiene las clases.
    top_n : int, optional
        Número de clases más frecuentes a conservar.

    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado con las columnas `objid`, `target_col` e
        `image_path`.
    """
    LOGGER.info(
        "Calculando las %d clases más frecuentes de la columna '%s' ...",
        top_n,
        target_col,
    )
    top_classes = merged_df[target_col].value_counts().nlargest(top_n).index.tolist()
    LOGGER.info("Clases seleccionadas: %s", top_classes)

    selected_cols = ["objid", target_col, "image_path"]
    filtered = merged_df[selected_cols].copy()
    filtered = filtered[filtered[target_col].isin(top_classes)]

    LOGGER.info("DataFrame filtrado con forma %s", filtered.shape)
    return filtered


def save_filtered_dataset(filtered_df: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda el DataFrame filtrado en CSV.

    Parameters
    ----------
    filtered_df : pandas.DataFrame
        DataFrame resultante tras el filtrado.
    output_path : Path
        Ruta de salida para el archivo CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    LOGGER.info("DataFrame filtrado guardado en %s", output_path)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada principal del script."""
    configure_logging()

    project_root = get_project_root()
    raw_dir, interim_dir = get_data_dirs(project_root)

    # 1. Descargar dataset de Kaggle
    dataset_name = "jaimetrickz/galaxy-zoo-2-images"
    dataset_path = download_kaggle_dataset(dataset_name)

    # Rutas esperadas dentro del dataset de Kaggle
    mapping_path = dataset_path / "gz2_filename_mapping.csv"
    images_root = dataset_path / "images_gz2" / "images"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de mapeo esperado: {mapping_path}. "
            "Revisa la estructura del dataset de Kaggle."
        )

    if not images_root.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de imágenes esperado: {images_root}. "
            "Revisa la estructura del dataset de Kaggle."
        )

    # 2. Descargar archivo de Hart 2016
    hart_url = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
    hart_gz_path = raw_dir / "gz2_hart16.csv.gz"
    download_hart_labels(hart_url, hart_gz_path)

    # 3. Cargar datos
    mapping_df = load_filename_mapping(mapping_path)
    hart_df = load_hart_dataframe(hart_gz_path)

    # 4. Unir mapping + etiquetas
    merged = merge_mapping_and_labels(mapping_df, hart_df)

    # 5. Añadir columna image_path
    merged_with_paths = add_image_paths(merged, images_root)

    # 6. Filtrar top 5 clases
    filtered_df = filter_top_classes(merged_with_paths, target_col="gz2_class", top_n=5)

    # 7. Guardar CSV final
    output_csv_path = interim_dir / "merged_filtered.csv"
    save_filtered_dataset(filtered_df, output_csv_path)

    LOGGER.info("Proceso de adquisición y preparación de datos completado correctamente.")


if __name__ == "__main__":
    main()
