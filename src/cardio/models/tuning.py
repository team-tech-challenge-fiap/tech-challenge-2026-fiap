"""Busca de hiperparâmetros (T14): uma função só, para todos os modelos serem ajustados do mesmo jeito.

Uso, com qualquer arquivo de ``cardio.models.estimators``::

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits("A")
    resultado = tune(modelo.build_model(X_train), modelo.PARAM_GRID, X_train, y_train, n_iter=20)
    resultado.best_model      # pipeline treinado no treino completo
    resultado.results         # um candidato por linha, com média e desvio de cada métrica

A busca usa **só o treino**: a validação cruzada de ``get_cv()`` roda dentro de ``X_train`` e o
pré-processador (passo ``"prep"``) é ajustado de novo em cada fold. A validação e o teste não
entram aqui; a validação é usada depois, para comparar os modelos já ajustados.
"""

import time
from typing import NamedTuple

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from ..config import SEED
from ..data.preprocessing import get_cv

SCORING = {
    "recall": "recall",
    "f1": "f1",
    "precision": "precision",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
}


class TuningResult(NamedTuple):
    """Saída de ``tune()``; pode ser desempacotada como tupla ou lida pelo nome do campo.

    Attributes:
        best_model: pipeline com os melhores parâmetros, treinado no treino **completo**.
        best_params: melhores valores do ``param_grid`` (chaves com o prefixo ``model__``).
        results: um candidato por linha, do melhor para o pior: uma coluna por parâmetro e,
            para cada métrica, ``<metrica>_media`` e ``<metrica>_desvio`` entre os folds.
        search_seconds: tempo da validação cruzada (todos os candidatos e folds).
        refit_seconds: tempo de treinar o melhor modelo no treino completo.
        n_search: número de pacientes usados na busca (menor que o treino se houve ``sample_size``).
    """

    best_model: Pipeline
    best_params: dict
    results: pd.DataFrame
    search_seconds: float
    refit_seconds: float
    n_search: int


def tune(
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    sample_size: int | None = None,
    refit: str = "roc_auc",
    n_jobs: int = -1,
    verbose: bool = True,
) -> TuningResult:
    """Busca aleatória de hiperparâmetros com validação cruzada estratificada no treino.

    Args:
        pipeline: saída de ``build_model()``, ainda sem treinar.
        param_grid: ``PARAM_GRID`` do modelo (listas de valores, com o prefixo ``model__``).
            Pode ser vazio: nesse caso só avalia o modelo como está.
        X_train, y_train: treino completo de ``load_splits()``.
        n_iter: número de combinações sorteadas; limitado ao tamanho do grid.
        sample_size: se informado (e menor que o treino), a busca roda numa amostra estratificada
            desse tamanho e o melhor modelo é treinado de novo no treino completo. Serve para
            modelos lentos, como SVM e KNN.
        refit: métrica que escolhe o melhor candidato. O padrão é a AUC, que não depende do limiar
            de decisão (ajustado depois, na T22) e funciona bem com as classes balanceadas.
        n_jobs: processos da busca. RF e XGBoost já usam vários núcleos por dentro; para eles pode
            valer ``n_jobs=1`` aqui, para os dois níveis não disputarem a mesma CPU.
        verbose: se True, imprime um resumo com as métricas e os tempos.

    Returns:
        ``TuningResult`` com o melhor modelo, os parâmetros, a tabela de candidatos e os tempos.

    Raises:
        ValueError: se ``refit``, ``n_iter`` ou ``sample_size`` forem inválidos.
    """
    if refit not in SCORING:
        raise ValueError(f"refit deve ser uma de {list(SCORING)}; recebido: {refit!r}")
    if n_iter < 1:
        raise ValueError(f"n_iter deve ser pelo menos 1; recebido: {n_iter}")
    if sample_size is not None and sample_size < 1:
        raise ValueError(f"sample_size deve ser pelo menos 1; recebido: {sample_size}")

    amostrar = sample_size is not None and sample_size < len(X_train)
    if amostrar:
        X_busca, _, y_busca, _ = train_test_split(
            X_train, y_train, train_size=sample_size, stratify=y_train, random_state=SEED
        )
    else:
        X_busca, y_busca = X_train, y_train

    candidatos = len(ParameterGrid(param_grid))  # 1 quando o grid é vazio
    busca = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=min(n_iter, candidatos),
        scoring=SCORING,
        refit=refit,
        cv=get_cv(),
        n_jobs=n_jobs,
        random_state=SEED,
        error_score="raise",  # uma falha não pode virar NaN em silêncio dentro de dezenas de ajustes
    )

    inicio = time.perf_counter()
    busca.fit(X_busca, y_busca)
    total = time.perf_counter() - inicio

    if amostrar:
        modelo = clone(pipeline).set_params(**busca.best_params_)
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        refit_seconds = time.perf_counter() - inicio
        search_seconds = total
    else:
        modelo = busca.best_estimator_
        refit_seconds = float(busca.refit_time_)
        search_seconds = total - refit_seconds

    resultado = TuningResult(
        best_model=modelo,
        best_params=busca.best_params_,
        results=_tabela_de_resultados(busca, refit),
        search_seconds=search_seconds,
        refit_seconds=refit_seconds,
        n_search=len(X_busca),
    )
    if verbose:
        _imprimir_resumo(resultado, candidatos=busca.n_iter, folds=busca.cv.get_n_splits(), refit=refit)
    return resultado


def _tabela_de_resultados(busca: RandomizedSearchCV, refit: str) -> pd.DataFrame:
    """Um candidato por linha: parâmetros e média/desvio de cada métrica, do melhor ao pior."""
    cv = busca.cv_results_
    metricas = {}
    for nome in SCORING:
        metricas[f"{nome}_media"] = cv[f"mean_test_{nome}"]
        metricas[f"{nome}_desvio"] = cv[f"std_test_{nome}"]
    tabela = pd.concat([pd.DataFrame(cv["params"]), pd.DataFrame(metricas)], axis=1)
    posicao = pd.Series(cv[f"rank_test_{refit}"], index=tabela.index)
    return tabela.loc[posicao.sort_values(kind="stable").index].reset_index(drop=True)


def _imprimir_resumo(resultado: TuningResult, candidatos: int, folds: int, refit: str) -> None:
    melhor = resultado.results.iloc[0]
    print(f"Busca: {candidatos} candidato(s) x {folds} folds em {resultado.n_search} pacientes "
          f"(escolha pela métrica '{refit}')")
    print(f"Melhores parâmetros: {resultado.best_params}")
    for nome in SCORING:
        print(f"  {nome}: {melhor[f'{nome}_media']:.4f} ± {melhor[f'{nome}_desvio']:.4f}")
    print(f"Tempo da busca: {resultado.search_seconds:.1f}s | "
          f"treino do melhor modelo: {resultado.refit_seconds:.1f}s")
