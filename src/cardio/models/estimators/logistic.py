"""T15 · Regressão Logística (baseline): contrato da T13 (``build_model()`` + ``PARAM_GRID``).

É o modelo mais simples e interpretável: cada coeficiente mostra quanto uma variável aumenta ou
diminui a chance de doença, a mesma lógica dos escores de risco clínicos. Serve de baseline: os
outros modelos precisam superá-lo para justificar a complexidade.

Hiperparâmetros padrão (escolhidos com validação cruzada de 5 folds no treino, ``get_cv()``):
o ``C`` quase não muda a AUC (0,792 no conjunto A e 0,794 no B, para C entre 0,01 e 10), e
``class_weight="balanced"`` aumenta um pouco o recall e o F1 sem perder AUC. Como em triagem o
falso negativo é o erro mais grave, ficou ``C=1`` com ``class_weight="balanced"``.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ...config import SEED
from ...data.preprocessing import build_preprocessor

PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__class_weight": [None, "balanced"],
}


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline da Regressão Logística, com as features contínuas escalonadas.

    Args:
        X_train: features de treino; só as colunas são usadas pelo pré-processador.

    Returns:
        ``Pipeline`` não treinado com os passos ``"prep"`` e ``"model"``.
    """
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)

    return Pipeline([
        ("prep", build_preprocessor(X_train, scale=True)),
        ("model", model),
    ])


def odds_ratios(pipeline: Pipeline) -> pd.DataFrame:
    """Converte os coeficientes de um pipeline treinado em odds ratio (``exp(coef)``).

    O odds ratio é a forma usada em medicina: 1,5 significa que a chance de doença fica 1,5 vez
    maior a cada aumento da variável; 0,8 significa que fica 20% menor.

    - Variáveis contínuas são escalonadas, então ``odds_ratio`` é **por desvio-padrão**;
      ``odds_ratio_por_unidade`` desfaz o escalonamento (ex.: por 1 mmHg, 1 ano, 1 kg).
    - Ordinais e binárias não são escalonadas: os dois valores são iguais (por nível ou sim/não).

    Args:
        pipeline: saída de ``build_model()`` já ajustada com ``fit``.

    Returns:
        DataFrame com ``variavel``, ``coeficiente``, ``odds_ratio`` e ``odds_ratio_por_unidade``,
        ordenado do maior para o menor efeito.
    """
    prep = pipeline.named_steps["prep"]
    coef = pipeline.named_steps["model"].coef_[0]
    nomes = prep.get_feature_names_out()

    # Desvio-padrão de cada contínua (as demais ficam com 1, porque não são escalonadas).
    desvios = pd.Series(1.0, index=nomes)
    scaler = prep.named_transformers_.get("num")
    if hasattr(scaler, "scale_"):
        desvios[list(scaler.get_feature_names_out())] = scaler.scale_

    tabela = pd.DataFrame({
        "variavel": nomes,
        "coeficiente": coef,
        "odds_ratio": np.exp(coef),
        "odds_ratio_por_unidade": np.exp(coef / desvios.to_numpy()),
    })
    return tabela.sort_values("odds_ratio", ascending=False, key=lambda s: abs(np.log(s))).reset_index(
        drop=True
    )
