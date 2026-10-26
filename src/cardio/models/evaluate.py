"""Avaliação padronizada dos modelos (T20).

Por que estas métricas (o problema é uma triagem de doença cardiovascular):

- **Recall**: de todos os pacientes doentes, quantos o modelo encontrou. É a métrica mais importante
  numa triagem, porque o falso negativo (doente mandado para casa) é o erro mais grave.
- **Precision**: de todos os que o modelo apontou como doentes, quantos realmente são. Mede o custo
  dos falsos positivos (exames e preocupação desnecessários).
- **F1**: média harmônica de precision e recall; resume o equilíbrio entre os dois num número só.
- **AUC (ROC)**: capacidade de separar doentes de saudáveis em qualquer limiar de decisão; não depende
  do corte de 0,5, então compara modelos de forma justa.
- **Accuracy**: também é informativa, porque as classes são balanceadas (~50% / 50%). Com o baseline
  em ~50%, ela mostra de imediato o ganho do modelo.

Referência: neste dataset (variáveis simples, e os hábitos são autodeclarados), uma acurácia em
torno de **73%** na validação é o esperado; valores muito acima disso merecem desconfiança (vazamento).
"""

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import FIGURES_DIR, SEED, VAL_RESULTS_PATH
from ..data.preprocessing import load_splits
from ..visualization.plots import plot_confusion_matrix

METRICAS = ["acuracia", "f1_score", "recall_score", "precision_score", "auc_roc"]  # colunas do val_results.csv
BASELINE = "baseline"


def evaluate(
    model,
    features,
    target,
    name: str | None = None,
    split: str = "val",
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, float]:
    """Calcula as métricas de classificação binária e, se ``name`` for dado, salva a matriz de confusão.

    Args:
        model: qualquer modelo/pipeline treinado com ``predict`` e ``predict_proba``.
        features, target: dados a avaliar (use a **validação**; o teste só na T23).
        name: nome do modelo. Quando informado, salva
            ``figures_dir / f"cm_{name}_{split}.png"`` (rótulos "Sem doença" e "Doença cardiovascular").
        split: ``"val"`` ou ``"test"``; entra no nome da figura e no título.
        figures_dir: pasta das figuras (``results/figures/`` por padrão).

    Returns:
        Dicionário com ``accuracy``, ``precision``, ``recall``, ``f1`` e ``roc_auc``.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    if name is not None:
        plot_confusion_matrix(
            target, predictions, Path(figures_dir) / f"cm_{name}_{split}.png",
            title=f"Matriz de confusão — {name} ({split})",
        )
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def evaluate_saved_model(
    model_path: Path, split: str = "val", figures_dir: Path = FIGURES_DIR
) -> dict[str, float]:
    """Avalia um modelo salvo em ``models/`` (``<modelo>_<A|B>.joblib``) na validação ou no teste.

    O conjunto de features (A ou B) vem do sufixo do nome do arquivo. Use ``split="test"`` só na T23.
    """
    model_path = Path(model_path)
    achou = re.fullmatch(r"(?P<modelo>.+)_(?P<conjunto>[AB])", model_path.stem)
    if achou is None:
        raise ValueError(f"O nome do arquivo deve terminar em _A ou _B (ex.: xgb_A.joblib): {model_path.name}")
    if split not in ("val", "test"):
        raise ValueError(f"split deve ser 'val' ou 'test'; recebido: {split!r}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(achou["conjunto"], save_ids=False)
    X, y = (X_val, y_val) if split == "val" else (X_test, y_test)
    return evaluate(joblib.load(model_path), X, y, name=model_path.stem, split=split, figures_dir=figures_dir)


def baseline_results(feature_set: str = "A") -> dict[str, float | str]:
    """Métricas do baseline na validação: ``DummyClassifier`` que sorteia a classe pelas proporções.

    É o piso de comparação (~50% de acurácia): qualquer modelo útil precisa ficar bem acima dele.
    """
    X_train, X_val, _, y_train, y_val, _ = load_splits(feature_set, save_ids=False)
    dummy = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_train, y_train)
    metricas = evaluate(dummy, X_val, y_val)
    return {
        "modelo": BASELINE,
        "features": feature_set,
        "acuracia": metricas["accuracy"],
        "f1_score": metricas["f1"],
        "recall_score": metricas["recall"],
        "precision_score": metricas["precision"],
        "auc_roc": metricas["roc_auc"],
    }


def summary_table(path: Path = VAL_RESULTS_PATH, include_baseline: bool = True) -> pd.DataFrame:
    """Junta os resultados de todos os modelos na validação, ordenados pela AUC (do melhor ao pior).

    Lê ``results/val_results.csv`` (uma linha por treino) e, se um modelo foi treinado mais de uma vez
    com o mesmo conjunto, mantém só o resultado mais recente. O baseline (``DummyClassifier``) entra
    sempre, para mostrar o ganho de cada modelo sobre o acaso.

    Args:
        path: CSV de resultados (colunas de ``METRICAS`` mais ``modelo`` e ``features``).
        include_baseline: se False, omite o baseline (útil em testes rápidos).

    Returns:
        DataFrame com ``modelo``, ``features`` e as métricas, e o índice reiniciado.
    """
    tabela = pd.read_csv(path)
    tabela = tabela[tabela["modelo"] != BASELINE].drop_duplicates(["modelo", "features"], keep="last")
    colunas = ["modelo", "features", *METRICAS]
    tabela = tabela[colunas]
    if include_baseline:
        tabela = pd.concat([tabela, pd.DataFrame([baseline_results()])[colunas]], ignore_index=True)
    return tabela.sort_values("auc_roc", ascending=False).reset_index(drop=True)
