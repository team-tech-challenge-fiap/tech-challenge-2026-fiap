"""Treinamento de um modelo e avaliação na validação (T13).

Uso: ``python -m cardio.models.train --modelo random_forest --features A [--sample 5000]``.
O nome do modelo é o de um arquivo em ``cardio.models.estimators``.
"""

import argparse
import time
from importlib import import_module
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import MODELS_DIR, SEED, VAL_RESULTS_PATH
from ..data.preprocessing import load_splits
from .tuning import tune


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento de modelos cardiovasculares")
    parser.add_argument("--modelo", type=str, required=True)
    parser.add_argument("--features", type=str, choices=["A", "B"], required=True)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Quantidade de amostras para testes rápidos",
    )
    parser.add_argument(
    "--tune",
    action="store_true",
    help="Ativa a busca de hiperparâmetros",
    )

    parser.add_argument(
    "--n-iter",
    type=int,
    default=20,
    help="Número de combinações na busca",
    )

    parser.add_argument(
    "--tune-sample",
    type=int,
    default=None,
    help="Quantidade de registros utilizados na busca",
    )

    args = parser.parse_args()
    args.modelo = args.modelo.lower()

    print("Modelo selecionado:", args.modelo)
    print("Conjunto de features:", args.features)

    nome_modulo = f"{__package__}.estimators.{args.modelo}"
    try:
        tipo_modelo = import_module(nome_modulo)
    except ModuleNotFoundError as erro:
        if erro.name == nome_modulo:
            parser.error(f"O modelo '{args.modelo}' ainda não foi implementado.")
        raise

    X_train, X_val, _, y_train, y_val, _ = load_splits(feature_set=args.features)  # teste só na T23

    if args.sample is not None:
        if args.sample <= 0 or args.sample > len(X_train):
            parser.error(f"--sample deve estar entre 1 e {len(X_train)}")
        X_train = X_train.sample(n=args.sample, random_state=SEED)
        y_train = y_train.loc[X_train.index]

    modelo = tipo_modelo.build_model(X_train)

    
    
    melhores_parametros = None
    tempo_busca = None

    print("Início treinamento.")
    inicio = time.perf_counter()

    if args.tune:
        if not hasattr(tipo_modelo, "PARAM_GRID") or not tipo_modelo.PARAM_GRID:
            parser.error(
                f"O modelo '{args.modelo}' não possui PARAM_GRID para otimização."
            )

        print("Iniciando busca de hiperparâmetros...")

        resultado_tuning = tune(
            pipeline=modelo,
            param_grid=tipo_modelo.PARAM_GRID,
            X_train=X_train,
            y_train=y_train,
            n_iter=args.n_iter,
            sample_size=args.tune_sample,
        )

        modelo = resultado_tuning.best_model
        melhores_parametros = resultado_tuning.best_params
        tempo_busca = resultado_tuning.search_seconds

        print("Melhores hiperparâmetros:", melhores_parametros)
        print(f"Tempo da busca: {tempo_busca:.2f} segundos")
        print(
            f"Tempo do reajuste: "
            f"{resultado_tuning.refit_seconds:.2f} segundos"
        )

    else:
        modelo.fit(X_train, y_train)

    tempo_treinamento = time.perf_counter() - inicio
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / f"{args.modelo}_{args.features}.joblib"
    joblib.dump(modelo, caminho_modelo)
    print(f"Modelo salvo em: {caminho_modelo.resolve()}")

    resultado = pd.DataFrame([{
        "modelo": args.modelo,
        "features": args.features,
        "acuracia": acuracia,
        "f1_score": f1,
        "recall_score": recall,
        "precision_score": precision,
        "auc_roc": roc,
        "tempo_treinamento": tempo_treinamento,
        "otimizacao": args.tune,
        "n_iter": args.n_iter if args.tune else None,
        "sample_size_busca": (
            args.tune_sample if args.tune else None
        ),
        "tempo_busca_segundos": tempo_busca,
        "melhores_parametros": (
            json.dumps(melhores_parametros)
            if melhores_parametros is not None
            else None
        ),
    }])

    VAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(
        VAL_RESULTS_PATH,
        mode="a",
        header=not VAL_RESULTS_PATH.exists(),
        index=False,
    )


if __name__ == "__main__":
    main()
