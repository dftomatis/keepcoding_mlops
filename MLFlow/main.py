# main.py

import argparse
import mlflow
import mlflow.sklearn
from funciones import cargar_dataset, vectorizar_textos, entrenar_modelo, evaluar_modelo

def main(args):
    # Cargar datos
    print("Cargando dataset...")
    train_texts, train_labels, test_texts, test_labels = cargar_dataset()

    # Vectorizar
    print("Vectorizando textos...")
    X_train, X_test, vectorizer = vectorizar_textos(train_texts, test_texts)

    # Entrenar
    print(f"Entrenando modelo: {args.model}")
    clf = entrenar_modelo(
        modelo=args.model,
        X_train=X_train,
        y_train=train_labels,
        C=args.C,
        n_estimators=args.n_estimators
    )

    # Evaluar
    print("Evaluando modelo...")
    accuracy, precision, recall, reporte = evaluar_modelo(clf, X_test, test_labels)

    print("\nClassification Report:")
    print(reporte)

    # Registrar con MLflow
    mlflow.set_experiment("SST2_Experimentos")

    with mlflow.start_run(run_name=f"{args.model}_run"):
        # Log de parámetros
        mlflow.log_param("model", args.model)
        mlflow.log_param("C", args.C)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("vectorizer", "TFIDF(max_features=5000, ngram_range=(1,2))")

        # Log de métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log del modelo
        mlflow.sklearn.log_model(clf, "model")

    print("Registro en MLflow completado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación de modelo SST-2 con MLflow.")
    parser.add_argument("--model", type=str, choices=["logreg", "randomforest"], default="logreg",
                        help="Modelo a entrenar: 'logreg' o 'randomforest'")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Valor del hiperparámetro C (solo para LogisticRegression)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Número de árboles (solo para RandomForest)")

    args = parser.parse_args()
    main(args)
