"""
Script de entrenamiento de modelos de clasificación de galaxias.

Uso desde la raíz del proyecto:

    python scripts/training/main.py --model baseline --epochs 5
    python scripts/training/main.py --model efficientnet --epochs 5

Modelos disponibles:
- baseline    : CNN sencilla definida en src/nombre_paquete/models/baseline.py
- efficientnet: EfficientNetB0 con transfer learning definida en
                src/nombre_paquete/models/efficientnet_b0.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import tensorflow as tf

# ---------------------------------------------------------------------
# Asegurar que la carpeta src/ esté en sys.path para poder importar
# el paquete nombre_paquete sin depender de PYTHONPATH externo.
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nombre_paquete.training.data_pipeline import (
    IMG_SIZE,
    create_datasets,
    get_project_root,
)
from nombre_paquete.models.baseline import build_baseline_cnn
from nombre_paquete.models.efficientnet_b0 import build_efficientnet_b0


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de galaxias.")
    parser.add_argument(
        "--model",
        choices=["baseline", "efficientnet"],
        default="baseline",
        help="Modelo a entrenar.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de lote.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proporción de datos para validación.",
    )
    return parser.parse_args()


def build_model(
    model_name: Literal["baseline", "efficientnet"],
    input_shape,
    num_classes: int,
) -> tf.keras.Model:
    """
    Construye el modelo correspondiente según el nombre indicado.
    """
    if model_name == "baseline":
        model = build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "efficientnet":
        # En primera instancia usamos la base congelada (train_base=False)
        model = build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            train_base=False,
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def main() -> None:
    """Punto de entrada principal del script."""
    configure_logging()
    args = parse_args()

    project_root = get_project_root()
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Creando datasets de entrenamiento y validación ...")
    train_ds, val_ds, class_to_index, index_to_class = create_datasets(
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    num_classes = len(class_to_index)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    LOGGER.info("Construyendo modelo '%s' ...", args.model)
    model = build_model(args.model, input_shape, num_classes)

    LOGGER.info("Compilando modelo ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks: early stopping y mejor modelo
    model_name = "baseline_cnn" if args.model == "baseline" else "efficientnet_b0"
    checkpoint_path = models_dir / f"{model_name}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    LOGGER.info("Iniciando entrenamiento por %d épocas ...", args.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    LOGGER.info("Evaluando modelo final en el conjunto de validación ...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

    # Guardar métricas y clases
    metrics = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
    }

    metrics_path = reports_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    LOGGER.info("Entrenamiento completado.")
    LOGGER.info("Modelo guardado en: %s", checkpoint_path)
    LOGGER.info("Métricas guardadas en: %s", metrics_path)


if __name__ == "__main__":
    main()
