"""Testes do ColumnTransformer de pré-processamento (T11)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cardio.data.preprocessing import build_preprocessor, load_splits
from cardio.features.build import (
    COLUNAS_BINARIAS,
    COLUNAS_CONTINUAS,
    COLUNAS_NOMINAIS,
    COLUNAS_ORDINAIS,
    FEATURES_B,
)


@pytest.fixture(scope="module")
def splits():
    return {nome: load_splits(nome, save_ids=False) for nome in ("A", "B")}


def test_splits_nao_carregam_o_log_da_limpeza_nos_attrs(splits) -> None:
    """Um DataFrame em ``attrs`` quebra o pd.concat do ColumnTransformer com ``scale=False``."""
    for X_train, X_val, X_test, y_train, y_val, y_test in splits.values():
        assert all(not parte.attrs for parte in (X_train, X_val, X_test, y_train, y_val, y_test))


def test_grupos_de_colunas_cobrem_o_conjunto_b_sem_repeticao() -> None:
    grupos = COLUNAS_CONTINUAS + COLUNAS_ORDINAIS + COLUNAS_BINARIAS + COLUNAS_NOMINAIS
    assert sorted(grupos) == sorted(FEATURES_B)


@pytest.mark.parametrize("conjunto", ["A", "B"])
def test_fit_transform_roda_sem_nan_nos_conjuntos_a_e_b(splits, conjunto) -> None:
    X_train, X_val, X_test, *_ = splits[conjunto]
    prep = build_preprocessor(X_train)
    treino = prep.fit_transform(X_train)

    assert isinstance(treino, pd.DataFrame)
    assert len(treino) == len(X_train)
    assert not treino.isna().any().any()
    for X in (X_val, X_test):
        assert not prep.transform(X).isna().any().any()

    # cada nominal vira uma coluna por categoria; as demais continuam com uma coluna cada
    nominais = [c for c in COLUNAS_NOMINAIS if c in X_train]
    esperado = X_train.shape[1] - len(nominais) + sum(X_train[c].nunique() for c in nominais)
    assert treino.shape[1] == esperado


def test_continuas_sao_padronizadas_so_com_o_treino(splits) -> None:
    X_train, X_val, *_ = splits["A"]
    prep = build_preprocessor(X_train).fit(X_train)

    treino, val = prep.transform(X_train), prep.transform(X_val)
    continuas = [c for c in COLUNAS_CONTINUAS if c in X_train]
    assert treino[continuas].mean().abs().max() < 1e-9
    assert np.allclose(treino[continuas].std(ddof=0), 1)
    # a validação usa a média e o desvio do treino, então não fica exatamente em 0 e 1
    assert not np.allclose(val[continuas].mean(), 0, atol=1e-9)


def test_ordinais_e_binarias_passam_sem_alteracao(splits) -> None:
    X_train, *_ = splits["B"]
    saida = build_preprocessor(X_train).fit_transform(X_train)
    for coluna in COLUNAS_ORDINAIS + COLUNAS_BINARIAS:
        assert saida[coluna].equals(X_train[coluna])
    assert set(saida["cholesterol"].unique()) == {1, 2, 3}


def test_scale_false_nao_escalona_as_continuas(splits) -> None:
    X_train, *_ = splits["B"]
    saida = build_preprocessor(X_train, scale=False).fit_transform(X_train)
    for coluna in COLUNAS_CONTINUAS:
        assert np.allclose(saida[coluna], X_train[coluna])


def test_categorias_nominais_viram_colunas_com_nomes_legiveis(splits) -> None:
    X_train, *_ = splits["B"]
    saida = build_preprocessor(X_train).fit_transform(X_train)
    assert "categoria_pressao_Hipertensão 2" in saida.columns
    assert "faixa_imc_Obesidade" in saida.columns
    assert not set(COLUNAS_NOMINAIS) & set(saida.columns)  # as originais foram substituídas


def test_categoria_nova_na_validacao_e_ignorada(splits) -> None:
    X_train, *_ = splits["B"]
    prep = build_preprocessor(X_train).fit(X_train)
    novo = X_train.head(3).copy()
    novo["categoria_pressao"] = "Crise hipertensiva"

    saida = prep.transform(novo)

    colunas = [c for c in saida.columns if c.startswith("categoria_pressao_")]
    assert (saida[colunas] == 0).all().all()


def test_coluna_desconhecida_levanta_erro(splits) -> None:
    X_train, *_ = splits["A"]
    with pytest.raises(ValueError, match="bmi"):
        build_preprocessor(X_train.assign(bmi=1.0))


def test_pipeline_com_preprocessador_treina_e_prediz_com_b(splits) -> None:
    X_train, X_val, _, y_train, *_ = splits["B"]
    pipeline = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=500)),
    ])
    pipeline.fit(X_train.head(2000), y_train.head(2000))
    assert pipeline.predict(X_val.head(10)).shape == (10,)
    assert len(pipeline[:-1].get_feature_names_out()) == pipeline[:-1].transform(X_val.head(1)).shape[1]
