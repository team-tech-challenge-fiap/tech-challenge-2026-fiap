
"""T23 — Comparação dos modelos na validação."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.dummy import DummyClassifier

from cardio.config import SEED, VAL_RESULTS_PATH
from cardio.data.preprocessing import load_splits
from cardio.models.evaluate import evaluate


OUTPUT_PATH = Path("results/comparison_table.csv")

COLUNAS = [
    "modelo",
    "features",
    "acuracia",
    "precision_score",
    "recall_score",
    "f1_score",
    "auc_roc",
    "tempo_treinamento",
]


def build_comparison_table() -> pd.DataFrame:
    """Compara os resultados mais recentes por modelo e conjunto."""

    df = pd.read_csv(VAL_RESULTS_PATH)

    # Mantém o resultado mais recente de cada modelo/conjunto.
    df = df[df["modelo"] != "baseline"].copy()

    df = df.drop_duplicates(
        subset=["modelo", "features"],
        keep="last",
    )

    # Calcula baseline separadamente para A e B.
    baselines = []

    for conjunto in ("A", "B"):
        X_train, X_val, _, y_train, y_val, _ = load_splits(
            conjunto,
            save_ids=False,
        )

        modelo_baseline = DummyClassifier(
            strategy="stratified",
            random_state=SEED,
        )

        inicio = perf_counter()
        modelo_baseline.fit(X_train, y_train)
        tempo = perf_counter() - inicio

        # Avalia na validação, sem acessar o teste.
        metricas = evaluate(
            modelo_baseline,
            X_val,
            y_val,
        )

        baselines.append({
            "modelo": "baseline",
            "features": conjunto,
            "acuracia": metricas["accuracy"],
            "precision_score": metricas["precision"],
            "recall_score": metricas["recall"],
            "f1_score": metricas["f1"],
            "auc_roc": metricas["roc_auc"],
            "tempo_treinamento": tempo,
        })

    df = pd.concat(
        [df, pd.DataFrame(baselines)],
        ignore_index=True,
    )

    thresholds = pd.read_csv("results/thresholds.csv")

    df = df.merge(
    thresholds[["modelo", "features", "limiar"]],
    on=["modelo", "features"],
    how="left",
    validate="one_to_one",
)

    df = df[COLUNAS + ["limiar"]]

    ajustadas = pd.read_csv("results/adjusted_metrics.csv")

    COLUNAS_AJUSTADAS = [
    "modelo",
    "features",
    "auc_roc_recalculada",
    "recall_ajustado",
    "precision_ajustada",
    "f1_ajustado",
    "doentes_nao_sinalizados_por_1000",
    "saudaveis_encaminhados_por_1000",
]

    df = df.merge(
    ajustadas[COLUNAS_AJUSTADAS],
    on=["modelo", "features"],
    how="left",
    validate="one_to_one",
    )

    df = df.sort_values(
        by=["auc_roc", "recall_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    tabela = build_comparison_table()
    print(tabela.to_string(index=False))
    print(f"\nTabela salva em: {OUTPUT_PATH}")
