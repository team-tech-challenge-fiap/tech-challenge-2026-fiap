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
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import FIGURES_DIR, SEED, THRESHOLD_GRID, THRESHOLD_MIN_RECALL, VAL_RESULTS_PATH
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


# --- Ajuste do limiar de decisão (T22) ---------------------------------------------------------
# Num sistema de apoio à triagem, o falso negativo (doente não sinalizado) é o erro mais grave.
# Baixar o limiar abaixo de 0,5 aumenta o recall, ao custo de mais falsos positivos.
# Regra: o MAIOR limiar que mantém o recall >= THRESHOLD_MIN_RECALL (0,80) na VALIDAÇÃO, ou seja,
# sinaliza pelo menos 8 em cada 10 doentes com o menor número possível de alarmes falsos.
# Nunca use o teste para escolher o limiar: ele é escolhido na validação e só aplicado no teste (T23).


def threshold_curve(y_true, probabilities, thresholds=None) -> pd.DataFrame:
    """Métricas de classificação para cada limiar de decisão (T22).

    Args:
        y_true: alvo real (0 = sem doença, 1 = doença), da **validação**.
        probabilities: ``predict_proba(X)[:, 1]`` do modelo, nas mesmas linhas de ``y_true``.
        thresholds: limiares a testar. Padrão: de 0,10 a 0,90 de 0,01 em 0,01 (``config.THRESHOLD_GRID``).

    Returns:
        DataFrame com uma linha por limiar: ``limiar``, ``recall``, ``precision``, ``f1``,
        ``especificidade`` e o impacto por 1.000 pacientes (``doentes_nao_sinalizados_por_1000`` e
        ``saudaveis_encaminhados_por_1000``).
    """
    if thresholds is None:
        inicio, fim, passo = THRESHOLD_GRID
        thresholds = np.round(np.arange(inicio, fim + passo / 2, passo), 2)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    linhas = [{"limiar": float(limiar), **_metricas_no_limiar(y_true, probabilities, limiar)}
              for limiar in thresholds]
    return pd.DataFrame(linhas)


def choose_threshold(y_true, probabilities, min_recall: float = THRESHOLD_MIN_RECALL, thresholds=None) -> float:
    """Escolhe o limiar de decisão: o **maior** limiar com recall >= ``min_recall`` na validação (T22).

    Entre os limiares que sinalizam pelo menos ``min_recall`` dos doentes, o maior é o que gera menos
    falsos positivos (encaminhamentos desnecessários).

    Args:
        y_true: alvo real da **validação** (nunca do teste).
        probabilities: ``predict_proba(X_val)[:, 1]`` do modelo.
        min_recall: recall mínimo exigido (padrão: ``config.THRESHOLD_MIN_RECALL`` = 0,80).
        thresholds: limiares candidatos (padrão: ``config.THRESHOLD_GRID``).

    Returns:
        O limiar escolhido, para usar como ``(proba >= limiar).astype(int)``.

    Raises:
        ValueError: se nenhum limiar candidato atingir o recall mínimo.
    """
    curva = threshold_curve(y_true, probabilities, thresholds)
    candidatos = curva[curva["recall"] >= min_recall]
    if candidatos.empty:
        raise ValueError(f"Nenhum limiar atinge recall >= {min_recall}; use limiares menores.")
    return float(candidatos["limiar"].max())


def impact_per_1000(y_true, probabilities, threshold: float) -> dict[str, float]:
    """Impacto de um limiar em números concretos, a cada 1.000 pacientes avaliados (T22).

    Returns:
        Dicionário com ``limiar``, ``recall``, ``precision``, ``f1``, ``especificidade``,
        ``doentes_nao_sinalizados_por_1000`` (falsos negativos) e
        ``saudaveis_encaminhados_por_1000`` (falsos positivos).
    """
    return {"limiar": float(threshold),
            **_metricas_no_limiar(np.asarray(y_true), np.asarray(probabilities), threshold)}


def _metricas_no_limiar(y_true: np.ndarray, probabilities: np.ndarray, limiar: float) -> dict[str, float]:
    previsto = (probabilities >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_true, previsto, labels=[0, 1]).ravel()
    total = len(y_true)
    return {
        "recall": float(recall_score(y_true, previsto, zero_division=0)),
        "precision": float(precision_score(y_true, previsto, zero_division=0)),
        "f1": float(f1_score(y_true, previsto, zero_division=0)),
        "especificidade": float(vn / (vn + fp)) if (vn + fp) else 0.0,
        "doentes_nao_sinalizados_por_1000": float(fn / total * 1000),
        "saudaveis_encaminhados_por_1000": float(fp / total * 1000),
    }
