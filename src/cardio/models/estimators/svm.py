"""T19 · SVM: contrato da T13 (``build_model()`` + ``PARAM_GRID``).

O SVM busca a fronteira que separa as classes com a maior margem possível. Com kernel RBF ele desenha
fronteiras curvas, mas o ``SVC`` exato fica muito lento com 48 mil pacientes (o custo cresce com o
quadrado do número de pacientes). Por isso foram comparadas três estratégias, com a busca de
hiperparâmetros de ``tune()`` (validação cruzada de 5 folds no treino, escolha pela AUC) e o resultado
na validação:

==============================================  ============  ============  ==============
Opção                                           AUC val. (A)  AUC val. (B)  Busca + treino
==============================================  ============  ============  ==============
1. ``LinearSVC`` calibrado (treino completo)    0,783         0,784         ~10 s + 1 s
2a. ``SVC(kernel="rbf")`` em 12 mil pacientes   0,783         –             ~320 s + 23 s
2b. ``Nystroem`` + ``LinearSVC`` calibrado      **0,792**     **0,791**     ~430 s + 30 s
==============================================  ============  ============  ==============

A opção escolhida é a **2b**: o ``Nystroem`` aproxima o kernel RBF com 500 componentes, e um
``LinearSVC`` treinado nesse espaço chega perto do SVM com kernel usando o treino inteiro, com custo
linear no número de pacientes. Ela é a única que supera a versão linear (que fica no nível da Regressão
Logística, ver T07: a pressão tem efeito de limiar que uma fronteira reta não captura) e também prevê
rápido (0,2 s para a validação inteira, contra 3 s do ``SVC`` exato). O ``SVC`` na amostra perde
justamente por ver só um quarto dos pacientes. O ``LinearSVC`` não dá probabilidades, então fica
dentro de ``CalibratedClassifierCV`` (calibração sigmoide com 5 folds), necessária para as curvas ROC
(T21) e o ajuste do limiar (T22).

Hiperparâmetros padrão: ``gamma=0.1`` e ``C=0.1``. No conjunto A foi a melhor combinação; no B a busca
escolheu ``gamma=0.05``, mas ``gamma=0.1`` empata na validação cruzada (AUC 0,7995 contra 0,7996),
então o mesmo padrão serve aos dois conjuntos. O ``C`` quase não muda a AUC entre 0,01 e 1.

O passo ``"model"`` é ele mesmo um ``Pipeline`` (``"rbf"`` + ``"svm"``), para o pipeline externo
manter só os passos ``"prep"`` e ``"model"`` do contrato.
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ...config import SEED
from ...data.preprocessing import build_preprocessor

N_COMPONENTES = 500  # dimensão da aproximação do kernel RBF

PARAM_GRID = {
    "model__rbf__gamma": [0.05, 0.1],
    "model__svm__estimator__C": [0.01, 0.1, 1],
}


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline do SVM com kernel RBF aproximado, com as features contínuas escalonadas.

    Args:
        X_train: features de treino; só as colunas são usadas pelo pré-processador.

    Returns:
        ``Pipeline`` não treinado com os passos ``"prep"`` e ``"model"``.
    """
    model = Pipeline([
        ("rbf", Nystroem(kernel="rbf", gamma=0.1, n_components=N_COMPONENTES, random_state=SEED)),
        ("svm", CalibratedClassifierCV(LinearSVC(C=0.1, random_state=SEED), cv=5)),
    ])

    return Pipeline([
        ("prep", build_preprocessor(X_train, scale=True)),
        ("model", model),
    ])
