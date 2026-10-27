"""Testes da feature importance (T24)."""

import numpy as np
import pytest

from cardio.data.preprocessing import load_splits
from cardio.models.estimators import knn, logistic, random_forest, xgb
from cardio.models.explain import (
    comparar_rankings,
    importancia_nativa,
    importancia_por_permutacao,
    odds_ratios_clinicos,
    plot_importancias,
    plot_odds_ratios,
    plot_ranking,
)

N_TREINO = 4000


@pytest.fixture(scope="module")
def dados():
    X_train, X_val, _, y_train, y_val, _ = load_splits("A", save_ids=False)
    return X_train.head(N_TREINO), y_train.head(N_TREINO), X_val.head(2000), y_val.head(2000)


@pytest.fixture(scope="module")
def modelos(dados):
    X, y, _, _ = dados
    return {
        "logistic": logistic.build_model(X).fit(X, y),
        "random_forest": random_forest.build_model(X).fit(X, y),
        "xgb": xgb.build_model(X).fit(X, y),
    }


@pytest.mark.parametrize("nome", ["random_forest", "xgb"])
def test_importancia_nativa_soma_um_e_tem_nomes_legiveis(modelos, nome) -> None:
    tabela = importancia_nativa(modelos[nome])

    assert len(tabela) == 11
    assert tabela["importancia"].sum() == pytest.approx(1)
    assert tabela["importancia"].is_monotonic_decreasing
    assert "Pressão sistólica (mmHg)" in set(tabela["variavel"])
    assert not {"ap_hi", "age_years"} & set(tabela["variavel"])  # nada de nome técnico


def test_importancia_nativa_recusa_modelo_sem_importancia_nativa(modelos) -> None:
    with pytest.raises(ValueError, match="importancia_por_permutacao"):
        importancia_nativa(modelos["logistic"])


def test_permutacao_devolve_todas_as_variaveis_e_a_pressao_vem_primeiro(dados, modelos) -> None:
    _, _, X_val, y_val = dados
    for nome, modelo in modelos.items():
        tabela = importancia_por_permutacao(modelo, X_val, y_val, n_repeats=3)

        assert set(tabela["variavel"]) == {"Idade (anos)", "Sexo masculino", "Altura (cm)", "Peso (kg)",
                                           "Pressão sistólica (mmHg)", "Pressão diastólica (mmHg)",
                                           "Colesterol (nível 1–3)", "Glicose (nível 1–3)", "Fumante",
                                           "Consome álcool", "Fisicamente ativo"}, nome
        assert tabela["queda_media"].is_monotonic_decreasing
        assert tabela["variavel"].iloc[0] == "Pressão sistólica (mmHg)", nome
        assert (tabela["desvio"] >= 0).all()


def test_permutacao_e_reprodutivel(dados, modelos) -> None:
    _, _, X_val, y_val = dados
    primeira = importancia_por_permutacao(modelos["xgb"], X_val, y_val, n_repeats=3)
    segunda = importancia_por_permutacao(modelos["xgb"], X_val, y_val, n_repeats=3)
    assert primeira.equals(segunda)


def test_permutacao_funciona_com_qualquer_modelo_e_com_o_conjunto_b() -> None:
    """KNN não tem importância nativa e o conjunto B tem colunas categóricas: a permutação cobre os dois."""
    X_train, X_val, _, y_train, y_val, _ = load_splits("B", save_ids=False)
    modelo = knn.build_model(X_train.head(800)).fit(X_train.head(800), y_train.head(800))

    tabela = importancia_por_permutacao(modelo, X_val.head(400), y_val.head(400), n_repeats=2)

    assert len(tabela) == 16  # uma linha por variável original, não por coluna one-hot
    assert {"IMC", "Pressão de pulso"} <= set(tabela["variavel"])


def test_odds_ratio_por_10_mmhg_e_a_exponencial_do_coeficiente(dados, modelos) -> None:
    pipeline = modelos["logistic"]
    tabela = odds_ratios_clinicos(pipeline).set_index("variavel")

    j = list(pipeline[:-1].get_feature_names_out()).index("ap_hi")
    escalonador = pipeline.named_steps["prep"].named_transformers_["num"]
    desvio_padrao = escalonador.scale_[list(escalonador.get_feature_names_out()).index("ap_hi")]
    coeficiente_por_mmhg = pipeline[-1].coef_[0][j] / desvio_padrao
    linha = tabela.loc["Pressão sistólica (mmHg)"]

    assert linha["incremento"] == "10 mmHg"
    assert linha["odds_ratio"] == pytest.approx(np.exp(10 * coeficiente_por_mmhg))
    assert linha["odds_ratio"] > 1  # pressão alta aumenta a chance de doença
    assert len(tabela) == 11


def test_comparar_rankings_ordena_pela_posicao_media() -> None:
    import pandas as pd

    ranking = comparar_rankings({
        "A": pd.Series({"x": 0.9, "y": 0.5, "z": 0.1}),
        "B": pd.Series({"x": 0.8, "y": 0.1, "z": 0.3}),
    })
    assert list(ranking.index) == ["x", "y", "z"]
    assert ranking.loc["z"].tolist() == [3, 2]


def test_graficos_sao_salvos(tmp_path, dados, modelos) -> None:
    _, _, X_val, y_val = dados
    permutacao = importancia_por_permutacao(modelos["xgb"], X_val, y_val, n_repeats=2)
    nativa = importancia_nativa(modelos["xgb"])
    ranking = comparar_rankings({"Permutação": permutacao.set_index("variavel")["queda_media"],
                                 "Nativa": nativa.set_index("variavel")["importancia"]})

    caminhos = [
        plot_importancias({"XGBoost": nativa}, "importancia", tmp_path / "nativa.png", "Nativa", "%", percentual=True),
        plot_importancias({"XGBoost": permutacao}, "queda_media", tmp_path / "perm.png", "Permutação",
                          "Queda de AUC", erro="desvio"),
        plot_odds_ratios(odds_ratios_clinicos(modelos["logistic"]), tmp_path / "or.png", "Odds ratio"),
        plot_ranking(ranking, tmp_path / "ranking.png", "Rankings"),
    ]
    assert all(caminho.exists() and caminho.stat().st_size > 5_000 for caminho in caminhos)
