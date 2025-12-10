"""
API de despliegue para el modelo de clasificación de galaxias.

Este script carga un modelo Keras entrenado y expone un servicio
REST usando FastAPI para predecir la clase morfológica de una galaxia
a partir de una imagen.

Uso recomendado (desde la raíz del proyecto):

    uvicorn scripts.evaluation.main:app --reload --host 0.0.0.0 --port 8000

El script asume que previamente se ha ejecutado el entrenamiento con:

    python scripts/training/main.py --model baseline --epochs N

lo cual genera:
    - models/baseline_cnn.keras
    - reports/baseline_cnn_metrics.json (incluye index_to_class)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

from nombre_paquete.training import IMG_SIZE


# ---------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------

# Este archivo está en: <root>/scripts/evaluation/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Detectar qué modelo usar: primero EfficientNet, si existe;
# si no, usar el baseline CNN (coherente con el notebook).
EFFICIENTNET_MODEL = MODELS_DIR / "efficientnet_b0.keras"
BASELINE_MODEL = MODELS_DIR / "baseline_cnn.keras"

if EFFICIENTNET_MODEL.exists():
    MODEL_NAME = "efficientnet_b0"
elif BASELINE_MODEL.exists():
    MODEL_NAME = "baseline_cnn"
else:
    raise RuntimeError(
        f"No se encontró ningún modelo entrenado en {MODELS_DIR}. "
        "Asegúrate de ejecutar primero `python scripts/training/main.py`."
    )

MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
METRICS_PATH = REPORTS_DIR / f"{MODEL_NAME}_metrics.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró el archivo de modelo: {MODEL_PATH}")

if not METRICS_PATH.exists():
    raise RuntimeError(f"No se encontró el archivo de métricas: {METRICS_PATH}")


# ---------------------------------------------------------------------
# Carga de modelo y mapeo de clases
# ---------------------------------------------------------------------

# Cargar modelo Keras
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar index_to_class desde el archivo de métricas generado por el entrenamiento
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

raw_index_to_class = metrics.get("index_to_class")
if raw_index_to_class is None:
    raise RuntimeError(
        "El archivo de métricas no contiene la clave 'index_to_class'. "
        "Revisa que el script de entrenamiento esté guardando esta información."
    )

# Aseguramos que las claves sean enteros
INDEX_TO_CLASS: Dict[int, str] = {int(k): v for k, v in raw_index_to_class.items()}


# ---------------------------------------------------------------------
# Definición de la API FastAPI
# ---------------------------------------------------------------------

app = FastAPI(
    title="API Clasificación de Galaxias",
    description="Servicio de inferencia para el modelo de clasificación de galaxias (Galaxy Zoo 2).",
    version="1.0.0",
)

# CORS sencillo (ajustar en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restringir en entornos productivos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    """
    Estructura de la respuesta de predicción.
    """
    predicted_class: str
    predicted_index: int
    probabilities: Dict[str, float]


# ---------------------------------------------------------------------
# Funciones auxiliares de preprocesamiento y predicción
# ---------------------------------------------------------------------

def _preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen para que sea compatible con el modelo:

    - Convierte a RGB
    - Redimensiona a IMG_SIZE (definido en nombre_paquete.training, p.ej. (128, 128))
    - Normaliza a [0, 1]
    - Añade dimensión de batch
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr


def _predict_from_bytes(image_bytes: bytes) -> PredictionResponse:
    """
    Realiza la predicción a partir de los bytes de una imagen.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.") from exc

    input_array = _preprocess_image(pil_img)

    # Inferencia
    preds = model.predict(input_array)
    probs = preds[0]

    predicted_index = int(np.argmax(probs))
    predicted_class = INDEX_TO_CLASS.get(predicted_index, str(predicted_index))

    # Construir diccionario de probabilidades por clase
    probs_by_class: Dict[str, float] = {}
    for idx, p in enumerate(probs):
        class_name = INDEX_TO_CLASS.get(idx, str(idx))
        probs_by_class[class_name] = float(p)

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_index=predicted_index,
        probabilities=probs_by_class,
    )


# ---------------------------------------------------------------------
# Endpoints de la API
# ---------------------------------------------------------------------

@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Endpoint de salud del servicio.
    """
    return {
        "status": "ok",
        "message": "API de clasificación de galaxias funcionando.",
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Endpoint de predicción: recibe una imagen (JPEG o PNG) y devuelve
    la clase predicha junto con las probabilidades por clase.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Formato no soportado. Solo se aceptan imágenes JPEG o PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    return _predict_from_bytes(image_bytes)
