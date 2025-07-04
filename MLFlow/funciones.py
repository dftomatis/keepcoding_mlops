# funciones.py

from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def cargar_dataset():
    """
    Descarga el dataset SST-2 y devuelve los datos de entrenamiento y validación.
    """
    dataset = load_dataset("glue", "sst2")
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["validation"]["sentence"]
    test_labels = dataset["validation"]["label"]
    return train_texts, train_labels, test_texts, test_labels

def vectorizar_textos(train_texts, test_texts, max_features=5000):
    """
    Vectoriza los textos usando TF-IDF.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def entrenar_modelo(modelo, X_train, y_train, C=1.0, n_estimators=100):
    """
    Entrena el modelo especificado.
    """
    if modelo == "logreg":
        clf = LogisticRegression(max_iter=1000, C=C)
    elif modelo == "randomforest":
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Modelo no soportado. Usa 'logreg' o 'randomforest'.")
    clf.fit(X_train, y_train)
    return clf

def evaluar_modelo(clf, X_test, y_test):
    """
    Calcula métricas de desempeño.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    return accuracy, precision, recall, reporte
