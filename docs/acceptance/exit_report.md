# Exit Report – Clasificación de Galaxias

## 1. Resumen ejecutivo

Este documento certifica el cierre exitoso del proyecto de clasificación morfológica de galaxias basado en redes neuronales convolucionales, desarrollado a partir del dataset Galaxy Zoo 2.

El proyecto cubre de manera integral el ciclo de vida de un sistema de Machine Learning, incluyendo la adquisición y preparación de datos, el entrenamiento y evaluación de modelos, el despliegue mediante una API REST y la validación funcional a través de inferencias reales sobre imágenes de galaxias.

---

## 2. Objetivos del proyecto

Los objetivos definidos al inicio del proyecto y su estado final se resumen a continuación:

| Objetivo | Estado |
|--------|--------|
| Preparar y filtrar el dataset Galaxy Zoo 2 | Cumplido |
| Entrenar un modelo CNN para clasificación morfológica | Cumplido |
| Evaluar el desempeño del modelo entrenado | Cumplido |
| Implementar un servicio de inferencia mediante API REST | Cumplido |
| Validar el sistema con predicciones reales | Cumplido |
| Documentar despliegue y criterios de aceptación | Cumplido |

---

## 3. Entrenamiento y evaluación del modelo

Se entrenó un modelo de red neuronal convolucional (CNN) baseline utilizando imágenes del dataset Galaxy Zoo 2, previamente filtradas y redimensionadas a un tamaño de 128×128 píxeles. Las imágenes fueron normalizadas en el rango [0, 1].

### Configuración principal del entrenamiento

- Tipo de modelo: CNN baseline
- Tamaño de imagen: 128 × 128
- Función de pérdida: Sparse Categorical Crossentropy
- Optimizador: Adam
- Número de épocas: 5

### Resultados obtenidos

- Accuracy de validación aproximada: 0.81
- Loss de validación aproximada: 0.55

### Artefactos generados

- Modelo entrenado:
  models/baseline_cnn.keras
- Métricas y mapeo de clases:
  reports/baseline_cnn_metrics.json

---

## 4. Despliegue del sistema

El modelo entrenado fue desplegado mediante una API REST desarrollada con FastAPI y ejecutada usando Uvicorn.

### Archivo principal de despliegue

scripts/evaluation/main.py

### Comando de ejecución

python -m uvicorn scripts.evaluation.main:app --reload --host 0.0.0.0 --port 8000

### Endpoints disponibles

- GET / – verificación del estado del servicio
- POST /predict – inferencia a partir de una imagen

La documentación interactiva de la API está disponible en:
http://127.0.0.1:8000/docs

---

## 5. Evidencia de funcionamiento

Ejemplo de respuesta del endpoint /predict:

{
  "predicted_class": "Ei",
  "predicted_index": 1,
  "probabilities": {
    "Ec": 0.018,
    "Ei": 0.717,
    "Er": 0.187,
    "Sc2t": 0.069,
    "Ser": 0.007
  }
}

La API respondió con código HTTP 200.

---

## 6. Criterios de aceptación

| Criterio | Resultado |
|--------|----------|
| El modelo entrena sin errores | Aprobado |
| El modelo genera métricas | Aprobado |
| La API inicia correctamente | Aprobado |
| Se realizan predicciones reales | Aprobado |
| El sistema es reproducible | Aprobado |

---

## 7. Limitaciones y trabajo futuro

- Mejorar el modelo con transfer learning.
- Contenerizar el servicio.
- Añadir autenticación y monitoreo.
- Optimizar tiempos de inferencia.

---

## 8. Declaración de cierre

Con base en los resultados obtenidos, el proyecto cumple con todos los criterios de aceptación y se declara formalmente cerrado.
