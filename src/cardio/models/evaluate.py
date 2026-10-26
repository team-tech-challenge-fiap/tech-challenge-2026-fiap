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

import pkgutil
import re
from importlib import import_module
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
from ..visualization.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
)
from . import estimators

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


# ---------------------------------------------------------------------------------------------
# Curvas ROC, Precision-Recall e calibração (T21), sempre na validação
# ---------------------------------------------------------------------------------------------


def available_models() -> list[str]:
    """Nomes dos modelos em ``cardio.models.estimators`` (menos o ``example``, que é só um teste do contrato).

    Quando um modelo novo (ex.: ``svm``, T19) entra na pasta, ele aparece aqui e nas curvas sozinho.
    """
    return sorted(m.name for m in pkgutil.iter_modules(estimators.__path__) if m.name != "example")


def validation_probabilities(feature_set: str = "A", models: list[str] | None = None) -> tuple[dict, pd.Series]:
    """Treina cada modelo com os hiperparâmetros padrão e devolve ``predict_proba`` na validação.

    Os modelos são treinados de novo (``SEED`` fixa, resultado reprodutível) em vez de lidos de
    ``models/``, para as curvas nunca usarem um ``.joblib`` antigo, de outra versão das features.

    Returns:
        ``({nome: probabilidade da classe 1}, y_val)``.
    """
    X_train, X_val, _, y_train, y_val, _ = load_splits(feature_set, save_ids=False)
    probabilidades = {}
    for nome in models or available_models():
        modelo = import_module(f"{estimators.__name__}.{nome}").build_model(X_train).fit(X_train, y_train)
        probabilidades[nome] = modelo.predict_proba(X_val)[:, 1]
    return probabilidades, y_val


def best_model_name(path: Path = VAL_RESULTS_PATH) -> str:
    """Modelo de maior AUC na validação (``summary_table`` sem o baseline).

    SUBSTITUIR T23: o modelo final é escolhido pelo Michel na T23; até lá, vale o melhor da validação.
    """
    return summary_table(path, include_baseline=False).iloc[0]["modelo"]


def make_curve_figures(figures_dir: Path = FIGURES_DIR, models: list[str] | None = None) -> dict[str, Path]:
    """Gera as figuras da T21 na validação e devolve ``{figura: caminho}``.

    - ``roc_A.png`` e ``pr_A.png``: todos os modelos no conjunto A, com AUC/AP na legenda e o baseline;
    - ``roc_A_vs_B_<melhor>.png``: o melhor modelo nos conjuntos A e B (as variáveis derivadas ajudam?);
    - ``calibration_<melhor>.png``: curva de calibração do melhor modelo (conjunto A).

    Interpretação (validação, os 5 modelos: knn, logistic, random_forest, svm e xgb; valores com a SEED do projeto):

    - ROC: as cinco curvas quase se sobrepõem (AUC de 0,783 na logística a 0,792 no SVM e no XGBoost) e ficam
      bem acima da diagonal do acaso (0,5): os modelos ordenam bem o risco, mas a escolha do modelo muda pouco
      o resultado. O SVM entra com as probabilidades calibradas pela própria T19.
    - Precision-Recall: com recall 0,6 a precision é ~0,8 em todos. XGBoost, Random Forest e SVM ficam juntos
      (AP de 0,774 a 0,776), acima do KNN (0,765) e da logística (0,761); todos bem acima do baseline (~0,5,
      a prevalência).
    - A vs. B: as curvas do conjunto A e B se sobrepõem (AUC 0,792 nos dois): as variáveis derivadas
      não acrescentam informação a um modelo de árvores.
    - Calibração: os pontos acompanham a diagonal; a probabilidade prevista pode ser lida como risco.
    - O "melhor modelo" (A vs. B e calibração) é o de maior AUC na validação; XGBoost (0,7917) e SVM (0,7915)
      empatam na prática, então pode mudar. É um substituto da T23: usar o modelo final quando ele existir.

    Args:
        figures_dir: pasta de saída (``results/figures/`` por padrão).
        models: nomes de ``available_models()`` a incluir; padrão: todos.
    """
    figures_dir = Path(figures_dir)
    proba_a, y_val = validation_probabilities("A", models)
    melhor = max(proba_a, key=lambda nome: roc_auc_score(y_val, proba_a[nome]))  # SUBSTITUIR T23: modelo final

    proba_b, y_val_b = validation_probabilities("B", [melhor])
    if not y_val.index.equals(y_val_b.index):  # A e B usam a mesma semente: devem ter os mesmos pacientes
        raise RuntimeError("A validação dos conjuntos A e B tem pacientes diferentes")

    return {
        "roc": plot_roc_curves(proba_a, y_val, figures_dir / "roc_A.png",
                               "Curvas ROC dos modelos — conjunto A (validação)"),
        "pr": plot_pr_curves(proba_a, y_val, figures_dir / "pr_A.png",
                             "Curvas Precision-Recall dos modelos — conjunto A (validação)"),
        "a_vs_b": plot_roc_curves(
            {f"{melhor} · A": proba_a[melhor], f"{melhor} · B": proba_b[melhor]}, y_val,
            figures_dir / f"roc_A_vs_B_{melhor}.png", f"{melhor}: conjunto A vs. B (validação)"),
        "calibration": plot_calibration(proba_a[melhor], y_val, figures_dir / f"calibration_{melhor}.png", melhor),
    }
