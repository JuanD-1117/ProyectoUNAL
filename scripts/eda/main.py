"""
Script de análisis exploratorio de datos (EDA) para el dataset filtrado.

Responsabilidades:
1. Cargar el archivo `data/interim/merged_filtered.csv`.
2. Mostrar en consola:
   - Forma del dataset.
   - Primeras filas.
   - Información de tipos de datos.
   - Distribución de la variable objetivo `gz2_class`.
3. Guardar un gráfico de barras con la distribución de clases en:
   `docs/data/figures/class_distribution.png`.

Este script está pensado para ejecutarse desde la raíz del proyecto:

    python scripts/eda/main.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configura el sistema de logging del módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def get_project_root() -> Path:
    """
    Devuelve la ruta a la raíz del proyecto.

    Se asume que este archivo está en:
        <root>/scripts/eda/main.py
    """
    return Path(__file__).resolve().parents[2]


def load_filtered_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Carga el archivo CSV con el dataset filtrado.

    Parameters
    ----------
    csv_path : Path
        Ruta al archivo `merged_filtered.csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame cargado.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset filtrado en {csv_path}. "
            "Asegúrate de ejecutar primero scripts/data_acquisition/main.py."
        )

    LOGGER.info("Cargando dataset filtrado desde %s ...", csv_path)
    df = pd.read_csv(csv_path)
    LOGGER.info("Dataset filtrado cargado con forma %s", df.shape)
    return df


def basic_eda(df: pd.DataFrame) -> None:
    """
    Ejecuta un análisis exploratorio básico y muestra resultados por consola.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con al menos las columnas `objid`, `gz2_class`, `image_path`.
    """
    LOGGER.info("Mostrando forma del dataset ...")
    print("\n=== Forma del dataset ===")
    print(df.shape)

    LOGGER.info("Mostrando primeras filas ...")
    print("\n=== Primeras filas ===")
    print(df.head())

    LOGGER.info("Mostrando información general del DataFrame ...")
    print("\n=== Información del DataFrame ===")
    print(df.info())

    LOGGER.info("Calculando distribución de la variable objetivo 'gz2_class' ...")
    class_counts = df["gz2_class"].value_counts()
    print("\n=== Distribución de clases (gz2_class) ===")
    print(class_counts)


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Genera y guarda un gráfico de barras con la distribución de clases.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna `gz2_class`.
    output_path : Path
        Ruta donde se guardará la imagen PNG.
    """
    LOGGER.info("Generando gráfico de distribución de clases ...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = df["gz2_class"].value_counts()

    plt.figure(figsize=(8, 5))
    class_counts.plot(kind="bar")
    plt.title("Distribución de clases (gz2_class)")
    plt.xlabel("Clase morfológica")
    plt.ylabel("Número de galaxias")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    LOGGER.info("Gráfico de distribución de clases guardado en %s", output_path)


def main() -> None:
    """Punto de entrada principal del script de EDA."""
    configure_logging()

    project_root = get_project_root()
    csv_path = project_root / "data" / "interim" / "merged_filtered.csv"
    figures_dir = project_root / "docs" / "data" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_filtered_dataset(csv_path)
    basic_eda(df)

    class_distribution_path = figures_dir / "class_distribution.png"
    plot_class_distribution(df, class_distribution_path)

    LOGGER.info("EDA básico completado correctamente.")


if __name__ == "__main__":
    main()
