# Proyecto de Machine Learning y MLOps con MLflow y FastAPI

Este repositorio contiene el desarrollo de una práctica completa de MLOps, organizada en dos grandes bloques:

- **Clasificación de Sentimientos con MLflow**: Entrenamiento y registro de modelos.
- **API REST con FastAPI**: Exposición de endpoints de inferencia y utilidades.

---

## Estructura del proyecto

```
/raiz
├── fastapi/
│   ├── Capturas_FastAPI - Swagger UI sin ejecutar.pdf
│   ├── Capturas_FastAPI - Swagger UI.pdf
│   ├── environment.yml
│   ├── playground.py
│   └── request.ipynb
│
├── MLFlow/
│   ├── Images/                 # Carpeta generada por MLflow para artefactos y modelos
│   ├── environment.yml         # Entorno Conda específico para MLflow
│   ├── funciones.py            # Funciones de preprocesamiento, entrenamiento y evaluación
│   ├── main.py                 # Script CLI principal
│   └── practica_mlops.ipynb    # Notebook exploratorio
│
└── README.md                   # Este archivo
```

---

## Parte 1: Clasificación de Sentimientos con MLflow

**Objetivo:**
Entrenar modelos de clasificación binaria sobre el dataset SST-2 y registrar experimentos con MLflow.

**Componentes principales:**
- `funciones.py`: contiene funciones reutilizables (vectorización, entrenamiento, métricas).
- `main.py`: permite entrenar modelos vía línea de comandos.
- `environment.yml`: define dependencias de Python y MLflow.
- `practica_mlops.ipynb`: notebook exploratorio que muestra todo el flujo paso a paso.
- Carpeta `Images/`: generada automáticamente por MLflow para almacenar artefactos (modelos, logs).

**Modelos implementados:**
- `LogisticRegression` con ajuste de hiperparámetro `C`.
- `RandomForestClassifier` con ajuste de `n_estimators`.

**Ejecución de la UI de MLflow:**

```
mlflow ui
```

Después, abre en tu navegador:
```
http://localhost:5000
```

---

## Parte 2: API REST con FastAPI

**Objetivo:**
Crear un servicio de API REST con distintos endpoints de prueba y dos pipelines de Hugging Face.

**Componentes principales:**
  - `playground.py`: script FastAPI con 5 endpoints:
  - `GET /saluda`: saludo dinámico.
  - `GET /cuadrado`: calcula el cuadrado de un número.
  - `GET /es_par`: determina si un número es par.
  - `POST /sentiment`: análisis de sentimiento con Hugging Face.
  - `POST /summary`: resumen de texto con Hugging Face.
  - `request.ipynb`: notebook con ejemplos de llamadas HTTP a los endpoints.
  - `environment.yml`: entorno Conda que incluye \`fastapi\`, \`transformers\` y dependencias de despliegue.
  - Capturas de pantalla:
    - `Capturas_FastAPI - Swagger UI sin ejecutar.pdf`: muestra la interfaz sin ejecutar peticiones.
    - `Capturas_FastAPI - Swagger UI.pdf`: muestra los endpoints probados desde Swagger.

**Ejecución del servidor:**

```
uvicorn playground:app --reload
```

Luego abre en el navegador:
```
http://127.0.0.1:8000/docs
```

**Nota:**
El despliegue en GCP Cloud Run no se realizó por falta de tiempo.

---

##  Requisitos Previos

- Anaconda o Miniconda instalado
- Python 3.10

---

##  Instalación de los entornos

### MLflow

```
cd MLFlow
conda env create -f environment.yml
conda activate sst2_mlflow_env
```

### FastAPI

```
cd fastapi
conda env create -f environment.yml
conda activate fastapi_env
```

---

##  Autor

Este proyecto fue realizado por el Ing. Darío Tomatis como práctica de Machine Learning y MLOps.

---

##  Notas finales

- El dataset SST-2 se descarga automáticamente desde `datasets`.
- Los modelos de Hugging Face también se descargan al primer uso.
- Todos los experimentos y artefactos quedan almacenados en MLflow.
