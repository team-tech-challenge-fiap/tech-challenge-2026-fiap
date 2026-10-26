"""T16 · KNN: contrato da T13 (``build_model()`` + ``PARAM_GRID``).

O KNN classifica um paciente pela maioria entre os ``k`` pacientes mais parecidos do treino. É fácil
de explicar ("pacientes parecidos tiveram o mesmo desfecho"), mas depende de distâncias: por isso as
variáveis contínuas são escalonadas (``scale=True``), senão o peso (kg) e a pressão (mmHg) dominariam
a idade (anos). Como o dataset é ruidoso, ``k`` precisa ser bem maior que o padrão (5), senão o modelo
decora os vizinhos mais próximos.

Hiperparâmetros padrão (escolhidos com ``tune()``: grid completo com validação cruzada de 5 folds numa
amostra de 10.000 pacientes do treino, escolha pela AUC): os conjuntos A e B escolheram
``n_neighbors=101`` e ``weights="uniform"`` (AUC de 0,782 e 0,779). A AUC cresce de forma contínua com
o ``k`` e o melhor valor está na borda do grid, então um ``k`` ainda maior pode render pouco mais; a
diferença para ``k=51`` é pequena (cerca de 0,002 de AUC), por isso o grid não foi estendido.

O KNN não aprende parâmetros: o ``fit`` só guarda o treino, e o custo está na predição, que compara
cada paciente com todos os do treino. Por isso é o modelo mais lento para prever.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ...data.preprocessing import build_preprocessor

PARAM_GRID = {
    "model__n_neighbors": [15, 31, 51, 101],
    "model__weights": ["uniform", "distance"],
}


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline do KNN, com as features contínuas escalonadas.

    Args:
        X_train: features de treino; só as colunas são usadas pelo pré-processador.

    Returns:
        ``Pipeline`` não treinado com os passos ``"prep"`` e ``"model"``.
    """
    model = KNeighborsClassifier(n_neighbors=101, weights="uniform", n_jobs=-1)

    return Pipeline([
        ("prep", build_preprocessor(X_train, scale=True)),
        ("model", model),
    ])
