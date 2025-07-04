# Proyecto de Clasificación de Sentimientos con SST-2, Scikit-Learn y MLflow

Este proyecto demuestra el ciclo completo de Machine Learning sobre el dataset SST-2 utilizando:

- Preprocesamiento de texto con TF-IDF.
- Modelos de clasificación (`LogisticRegression` y `RandomForestClassifier`).
- Registro de experimentos con MLflow.

---

## 📂 Estructura del proyecto

```
proyecto/
├── funciones.py           # Funciones reutilizables de carga, vectorización, entrenamiento y evaluación
├── main.py                # Script principal con argumentos CLI
├── environment.yml        # Definición del entorno Conda
├── README.md              # Este archivo
└── notebook.ipynb         # Notebook exploratorio (opcional)
```

---

## ⚙️ Requisitos previos

- Anaconda o Miniconda instalado
- Python 3.10

---

## ✅ Instalación del entorno

Crea y activa el entorno con:

```bash
conda env create -f environment.yml
conda activate sst2_mlflow_env
```

---

## 🚀 Ejecución del proyecto

Desde la terminal, con el entorno activado:

### Entrenar un modelo Logistic Regression:

```bash
python main.py --model logreg --C 0.5
```

### Entrenar un modelo Random Forest:

```bash
python main.py --model randomforest --n_estimators 200
```

---

## 🧩 Argumentos de `main.py`

| Argumento        | Descripción                                               | Valores posibles                |
|------------------|-----------------------------------------------------------|---------------------------------|
| `--model`        | Modelo a entrenar                                         | `logreg` o `randomforest`       |
| `--C`            | Hiperparámetro C de regularización (solo LogisticRegression) | Ej: 0.5                         |
| `--n_estimators` | Número de árboles (solo RandomForest)                     | Ej: 100                         |

---

## 📝 Registro con MLflow

El script automáticamente:

- Registra métricas (accuracy, precision, recall)
- Registra hiperparámetros
- Guarda el modelo entrenado

Para visualizar la UI de MLflow:

```bash
mlflow ui
```

Luego abre en tu navegador:

```
http://localhost:5000
```

---

## ✨ Notas

- El dataset SST-2 se descarga automáticamente con `datasets`.
- Puedes personalizar el vectorizador TF-IDF modificando `funciones.py`.
- Todas las ejecuciones quedan almacenadas en el tracking de MLflow.

---

## 📧 Autor

Este proyecto fue realizado por el Ing. Darío Tomatis como práctica de Machine Learning y MLOps.

---
