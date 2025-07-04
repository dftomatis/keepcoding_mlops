# playground.py
from fastapi import FastAPI
from transformers import pipeline

# Crear la app
app = FastAPI()

# Hugging Face pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
summarizer_pipeline = pipeline("summarization")

# Endpoint 1: Saludo
@app.get("/saluda")
def saluda(nombre: str = "Mundo"):
    return {"mensaje": f"Hola, {nombre}!"}

# Endpoint 2: Calcular cuadrado
@app.get("/cuadrado")
def cuadrado(numero: int):
    return {"resultado": numero ** 2}

# Endpoint 3: Es par
@app.get("/es_par")
def es_par(numero: int):
    return {"es_par": numero % 2 == 0}

# Endpoint 4: Análisis de sentimiento (Hugging Face)
@app.post("/sentiment")
def sentiment(data: dict):
    resultado = sentiment_pipeline(data["text"])
    return {"sentiment": resultado}

# Endpoint 5: Resumen de texto (Hugging Face)
@app.post("/summary")
def summary(data: dict):
    resumen = summarizer_pipeline(data["text"], max_length=50, min_length=10, do_sample=False)
    return {"summary": resumen}