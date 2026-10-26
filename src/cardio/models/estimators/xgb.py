"""T18 · XGBoost: contrato da T13 (``build_model()`` + ``PARAM_GRID``).

Gradient boosting costuma ser o melhor modelo em dados tabulares desse tamanho e funciona bem com o
SHAP (T25). Ele constrói as árvores em sequência, cada uma corrigindo os erros das anteriores, então
precisa de profundidade e taxa de aprendizado baixas para não decorar o ruído do dataset.

Hiperparâmetros padrão (escolhidos com ``tune()``: 20 candidatos do ``PARAM_GRID`` com validação
cruzada de 5 folds no treino, escolha pela AUC): os conjuntos A e B escolheram a mesma combinação,
com AUC de 0,801 nos dois. As 8 melhores combinações ficam entre 0,8005 e 0,8012, e todas usam
``learning_rate`` baixo e árvores rasas (profundidade 3 a 6), então o resultado é estável.

O arquivo se chama ``xgb.py`` (e não ``xgboost.py``) para não conflitar com a biblioteca.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ...config import SEED
from ...data.preprocessing import build_preprocessor

PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 4, 6],
    "model__learning_rate": [0.03, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline do XGBoost, sem escalonamento (modelos de árvore não precisam).

    Args:
        X_train: features de treino; só as colunas são usadas pelo pré-processador.

    Returns:
        ``Pipeline`` não treinado com os passos ``"prep"`` e ``"model"``.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )

    return Pipeline([
        ("prep", build_preprocessor(X_train, scale=False)),
        ("model", model),
    ])
