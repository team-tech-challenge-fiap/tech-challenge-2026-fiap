
"""T23 — Métricas dos modelos com os limiares definidos na T22.

Usa exclusivamente treino e validação. Não avalia o conjunto de teste.
"""

from importlib import import_module
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from cardio.data.preprocessing import load_splits
from cardio.models.evaluate import impact_per_1000


THRESHOLDS_PATH = Path("results/thresholds.csv")
OUTPUT_PATH = Path("results/adjusted_metrics.csv")


def main():
    thresholds = pd.read_csv(THRESHOLDS_PATH)
    resultados = []

    for conjunto in ("A", "B"):
        X_train, X_val, _, y_train, y_val, _ = load_splits(
            conjunto, save_ids=False
        )

        for nome in ("xgb", "random_forest", "knn", "logistic"):
            print(f"Processando {nome} - {conjunto}...", flush=True)

            modulo = import_module(f"cardio.models.estimators.{nome}")
            modelo = modulo.build_model(X_train)

            # Reproduz os hiperparâmetros ajustados da T17.
            if nome == "random_forest":
                if conjunto == "A":
                    modelo.set_params(
                        model__n_estimators=300,
                        model__max_depth=10,
                        model__min_samples_leaf=20,
                        model__max_features="sqrt",
                    )
                else:
                    modelo.set_params(
                        model__n_estimators=300,
                        model__max_depth=10,
                        model__min_samples_leaf=50,
                        model__max_features=0.5,
                    )

            modelo.fit(X_train, y_train)
            probabilidades = modelo.predict_proba(X_val)[:, 1]

            registro = thresholds.loc[
                (thresholds["modelo"] == nome)
                & (thresholds["features"] == conjunto)
            ]

            if len(registro) != 1:
                raise ValueError(
                    f"Limiar ausente ou duplicado: {nome} - {conjunto}"
                )

            limiar = float(registro["limiar"].iloc[0])
            metricas = impact_per_1000(y_val, probabilidades, limiar)

            resultados.append({
                "modelo": nome,
                "features": conjunto,
                "limiar": limiar,
                "auc_roc_recalculada": roc_auc_score(
                    y_val, probabilidades
                ),
                "recall_ajustado": metricas["recall"],
                "precision_ajustada": metricas["precision"],
                "f1_ajustado": metricas["f1"],
                "doentes_nao_sinalizados_por_1000":
                    metricas["doentes_nao_sinalizados_por_1000"],
                "saudaveis_encaminhados_por_1000":
                    metricas["saudaveis_encaminhados_por_1000"],
            })

    tabela = pd.DataFrame(resultados)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(OUTPUT_PATH, index=False)

    print("\n=== MÉTRICAS AJUSTADAS ===")
    print(tabela.to_string(index=False))
    print(f"\nArquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
