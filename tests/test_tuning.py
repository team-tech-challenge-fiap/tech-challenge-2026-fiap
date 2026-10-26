"""Testes da busca de hiperparâmetros (T14)."""

import pytest
from sklearn.pipeline import Pipeline

from cardio.data.preprocessing import load_splits
from cardio.models.estimators import example, logistic, xgb
from cardio.models.tuning import SCORING, TuningResult, tune


@pytest.fixture(scope="module")
def treino():
    X_train, _, _, y_train, *_ = load_splits("A", save_ids=False)
    return X_train.head(1500), y_train.head(1500)


def test_tune_com_o_modelo_de_exemplo_devolve_metricas_e_tempo(treino) -> None:
    X, y = treino
    resultado = tune(example.build_model(X), example.PARAM_GRID, X, y, verbose=False)

    assert isinstance(resultado, TuningResult)
    assert isinstance(resultado.best_model, Pipeline)
    assert resultado.best_params == {}
    assert len(resultado.results) == 1  # grid vazio: só avalia o modelo como está
    for metrica in SCORING:
        assert {f"{metrica}_media", f"{metrica}_desvio"} <= set(resultado.results.columns)
    assert resultado.search_seconds > 0 and resultado.refit_seconds > 0
    assert resultado.n_search == len(X)


def test_tune_com_sample_size_busca_na_amostra_e_treina_no_treino_completo(treino) -> None:
    X, y = treino
    resultado = tune(logistic.build_model(X), logistic.PARAM_GRID, X, y, n_iter=3, sample_size=400,
                     verbose=False)

    assert resultado.n_search == 400
    # o melhor modelo foi re-treinado com todos os pacientes, não só com a amostra
    assert resultado.best_model.named_steps["prep"].named_transformers_["num"].n_samples_seen_ == len(X)
    assert resultado.best_model.predict(X.head(5)).shape == (5,)


def test_tune_roda_com_o_grid_do_xgboost(treino) -> None:
    X, y = treino
    resultado = tune(xgb.build_model(X), xgb.PARAM_GRID, X, y, n_iter=2, sample_size=500, n_jobs=1,
                     verbose=False)
    assert set(resultado.best_params) == set(xgb.PARAM_GRID)
    assert resultado.best_model.predict(X.head(5)).shape == (5,)


def test_sample_size_maior_que_o_treino_nao_amostra(treino) -> None:
    X, y = treino
    resultado = tune(example.build_model(X), example.PARAM_GRID, X, y, sample_size=10 * len(X),
                     verbose=False)
    assert resultado.n_search == len(X)


def test_tune_escolhe_o_melhor_candidato_dentro_do_grid(treino) -> None:
    X, y = treino
    resultado = tune(logistic.build_model(X), logistic.PARAM_GRID, X, y, n_iter=4, verbose=False)

    assert len(resultado.results) == 4
    assert set(resultado.best_params) == set(logistic.PARAM_GRID)
    for parametro, valor in resultado.best_params.items():
        assert valor in logistic.PARAM_GRID[parametro]
    # a tabela vem do melhor para o pior pela métrica de escolha (roc_auc)
    aucs = resultado.results["roc_auc_media"]
    assert aucs.is_monotonic_decreasing
    melhor = resultado.results.iloc[0]
    for parametro, valor in resultado.best_params.items():
        assert melhor[parametro] == valor


def test_n_iter_maior_que_o_grid_e_limitado_ao_tamanho_do_grid(treino) -> None:
    X, y = treino
    grid = {"model__C": [0.1, 1]}
    resultado = tune(logistic.build_model(X), grid, X, y, n_iter=50, verbose=False)
    assert len(resultado.results) == 2


def test_tune_e_reprodutivel(treino) -> None:
    X, y = treino
    primeiro = tune(logistic.build_model(X), logistic.PARAM_GRID, X, y, n_iter=3, verbose=False)
    segundo = tune(logistic.build_model(X), logistic.PARAM_GRID, X, y, n_iter=3, verbose=False)
    assert primeiro.best_params == segundo.best_params
    assert primeiro.results["roc_auc_media"].tolist() == segundo.results["roc_auc_media"].tolist()


def test_tune_nao_modifica_o_pipeline_recebido(treino) -> None:
    X, y = treino
    pipeline = logistic.build_model(X)
    tune(pipeline, logistic.PARAM_GRID, X, y, n_iter=2, verbose=False)
    assert not hasattr(pipeline.named_steps["model"], "coef_")  # continua sem treino


def test_tune_imprime_o_resumo(treino, capsys) -> None:
    X, y = treino
    tune(example.build_model(X), example.PARAM_GRID, X, y)
    saida = capsys.readouterr().out
    assert "roc_auc" in saida and "Tempo da busca" in saida


@pytest.mark.parametrize("argumentos", [{"refit": "mcc"}, {"n_iter": 0}, {"sample_size": 0}])
def test_argumentos_invalidos_levantam_erro(treino, argumentos) -> None:
    X, y = treino
    with pytest.raises(ValueError):
        tune(example.build_model(X), example.PARAM_GRID, X, y, verbose=False, **argumentos)
