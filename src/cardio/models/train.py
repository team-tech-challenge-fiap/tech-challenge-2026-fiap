"""Treinamento do modelo baseline."""


import argparse
import joblib
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
#from cardio.preprocessing import load_splits
from cardio.data.preprocessing import load_splits
from cardio.models import example
from pathlib import Path
from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento de modelos cardiovasculares")

    parser.add_argument(
        "--modelo",
        type=str,
        required=True
    )

    parser.add_argument(
        "--features",
        type=str,
        choices=["A", "B"],
        required=True
    )

    parser.add_argument(
    "--sample",
    type=int,
    default=None,
    help="Quantidade de amostras para testes rápidos"
    )

    args = parser.parse_args()
    args.modelo = args.modelo.lower()


    print("Modelo selecionado:", args.modelo)
    print("Conjunto de features:", args.features)

    modelos = {
    "example", "logistic", "knn",
    "random_forest", "svm"
    }

    try:
    
        tipo_modelo = import_module(f"cardio.models.{args.modelo}")
    
    except ModuleNotFoundError as erro:
        if erro.name == f"cardio.models.{args.modelo}":
            parser.error(f"O modelo '{args.modelo}' ainda não foi implementado.")
        raise

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(
    feature_set=args.features
    )

    if args.sample is not None:
        if args.sample <= 0 or args.sample > len(X_train):
            parser.error(
            f"--sample deve estar entre 1 e {len(X_train)}"
        )

        X_train = X_train.sample(
        n=args.sample,
        random_state=42
    )

        y_train = y_train.loc[X_train.index]


    modelo = tipo_modelo.build_model(X_train)
    
    print("Início treinamento.")
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treinamento =  time.perf_counter() - inicio
    print("Fim do treinamento.")

    acuracia_treino = modelo.score(X_train, y_train)

    print(f"Acurácia no treinamento: {acuracia_treino:.2%}")


    y_pred = modelo.predict(X_val)
    y_proba = modelo.predict_proba(X_val)[:, 1]
      
    acuracia = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    confmx = confusion_matrix(y_val, y_pred, labels=[0, 1])
    roc = roc_auc_score(y_val, y_proba)
    

    print(f"Acurácia na validação: {acuracia:.2%}")
    print(f"F1-score validação: {f1:.2%}")
    print(f"Recall validação: {recall:.2%}")
    print(f"Precision validação: {precision:.2%}")
    print(confmx)
    print(f"Pacientes utilizados no treinamento: {len(X_train)}")
    print(f"Respostas utilizadas no treinamento: {len(y_train)}")
    print(f"AUC ROC validação: {roc:.2%}")

    vn, fp, fn, vp = confmx.ravel()

    print(f"Verdadeiros negativos: {vn}")
    print(f"Falsos positivos: {fp}")
    print(f"Falsos negativos: {fn}")
    print(f"Verdadeiros positivos: {vp}")


    caminho_csv = Path("results/val_results.csv")
    pasta_modelos = Path("models")
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_modelos / f"{args.modelo}_{args.features}.joblib"

    joblib.dump(modelo, caminho_modelo)

    print(f"Modelo salvo em: {caminho_modelo.resolve()}")

    resultado = pd.DataFrame([{
    "modelo": args.modelo,
    "features": args.features,
    "acuracia": acuracia,
    "f1_score": f1,
    "recall_score": recall,
    "precision_score": precision,
    "f1_score": f1,
    "auc_roc": roc,
    "tempo_treinamento": tempo_treinamento
    }])

    resultado.to_csv(
    caminho_csv,
    mode="a",
    header=not caminho_csv.exists(),
    index=False
    )

if __name__ == "__main__":
    
    main()
