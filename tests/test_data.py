"""Testes de qualidade dos dados e de vazamento (T12).

Garantem que os dados e o split continuem corretos à medida que o código muda:

1. ``id`` e ``cardio`` nunca aparecem nas features;
2. depois de ``clean()``, não sobram pressões, alturas ou pesos fora das faixas;
3. treino, validação e teste não têm pacientes em comum;
4. o split é reprodutível e estratificado (cada conjunto com 50% ± 1% de cada classe);
5. o pré-processador é ajustado só no treino.

Rodar com ``pytest tests/test_data.py`` (ou ``docker compose run --rm tests``, que roda todos os testes).
"""

import numpy as np
import pytest

from cardio.config import (
    DATA_PATH,
    PLAUSIBLE_RANGES,
    PRESSURE_RANGES,
    TARGET,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from cardio.data.loader import load_dataset, split_features_target
from cardio.data.preprocessing import build_preprocessor, clean, load_splits
from cardio.features.build import COLUNAS_CONTINUAS
from cardio.models.estimators import logistic

CONJUNTOS = ["A", "B"]
PROIBIDAS = {"id", TARGET}


@pytest.fixture(scope="module")
def splits():
    return {nome: load_splits(nome, save_ids=False) for nome in CONJUNTOS}


@pytest.fixture(scope="module")
def limpo():
    return clean(load_dataset(DATA_PATH), verbose=False)


# --- Teste 1: id e cardio nunca aparecem nas features ---

@pytest.mark.parametrize("conjunto", CONJUNTOS)
def test_id_e_alvo_nunca_aparecem_nas_features(splits, conjunto) -> None:
    X_train, X_val, X_test, *_ = splits[conjunto]
    for X in (X_train, X_val, X_test):
        assert not PROIBIDAS & set(X.columns)


@pytest.mark.parametrize("conjunto", CONJUNTOS)
def test_id_e_alvo_nao_saem_do_preprocessador(splits, conjunto) -> None:
    X_train, *_ = splits[conjunto]
    saida = build_preprocessor(X_train).fit(X_train).get_feature_names_out()
    assert not PROIBIDAS & set(saida)


def test_split_features_target_remove_id_e_alvo(limpo) -> None:
    X, y = split_features_target(limpo)
    assert not PROIBIDAS & set(X.columns)
    assert y.name == TARGET


# --- Teste 2: depois de clean(), tudo dentro das faixas ---

@pytest.mark.parametrize("coluna", ["ap_hi", "ap_lo", "height", "weight"])
def test_clean_deixa_cada_coluna_dentro_da_faixa(limpo, coluna) -> None:
    minimo, maximo = {**PRESSURE_RANGES, **PLAUSIBLE_RANGES}[coluna]
    fora = limpo[~limpo[coluna].between(minimo, maximo)]
    assert fora.empty, f"{len(fora)} pacientes com {coluna} fora de {minimo}–{maximo}"


def test_clean_nao_deixa_diastolica_maior_ou_igual_a_sistolica(limpo) -> None:
    assert (limpo["ap_lo"] < limpo["ap_hi"]).all()


def test_clean_nao_deixa_valores_vazios(limpo) -> None:
    assert not limpo.isna().any().any()


# --- Teste 3: nenhum paciente em mais de um conjunto ---

@pytest.mark.parametrize("conjunto", CONJUNTOS)
def test_treino_validacao_e_teste_nao_tem_pacientes_em_comum(splits, conjunto) -> None:
    X_train, X_val, X_test, *_ = splits[conjunto]
    treino, val, teste = set(X_train.index), set(X_val.index), set(X_test.index)

    assert not treino & val
    assert not treino & teste
    assert not val & teste
    for X in (X_train, X_val, X_test):
        assert X.index.is_unique


def test_os_tres_conjuntos_cobrem_o_dataset_limpo(splits, limpo) -> None:
    X_train, X_val, X_test, *_ = splits["A"]
    ids = X_train.index.union(X_val.index).union(X_test.index)
    assert ids.sort_values().tolist() == sorted(limpo["id"])


# --- Teste 4: split reprodutível, 70/15/15 e estratificado ---

@pytest.mark.parametrize("conjunto", CONJUNTOS)
def test_split_e_reprodutivel(splits, conjunto) -> None:
    de_novo = load_splits(conjunto, save_ids=False)
    for original, repetido in zip(splits[conjunto], de_novo):
        assert original.index.equals(repetido.index)


def test_conjuntos_a_e_b_usam_os_mesmos_pacientes(splits) -> None:
    for parte_a, parte_b in zip(splits["A"][3:], splits["B"][3:]):
        assert parte_a.index.equals(parte_b.index)


def test_split_tem_proporcao_70_15_15(splits) -> None:
    y_train, y_val, y_test = splits["A"][3:]
    total = len(y_train) + len(y_val) + len(y_test)
    for y, esperado in ((y_train, TRAIN_SIZE), (y_val, VAL_SIZE), (y_test, TEST_SIZE)):
        assert len(y) / total == pytest.approx(esperado, abs=0.001)


def test_split_e_estratificado(splits, limpo) -> None:
    taxa_geral = limpo[TARGET].mean()
    for y in splits["A"][3:]:
        taxa = y.mean()
        assert taxa == pytest.approx(0.5, abs=0.01)  # 50% ± 1% de cada classe
        assert taxa == pytest.approx(taxa_geral, abs=0.001)  # mesma taxa do dataset limpo


# --- Teste 5: o pré-processador é ajustado só no treino ---

def test_scaler_do_pipeline_usa_so_a_media_do_treino(splits, limpo) -> None:
    X_train, _, _, y_train, *_ = splits["A"]
    pipeline = logistic.build_model(X_train).fit(X_train, y_train)

    scaler = pipeline.named_steps["prep"].named_transformers_["num"]
    continuas = list(scaler.get_feature_names_out())
    assert set(continuas) <= set(COLUNAS_CONTINUAS)

    media_treino = X_train[continuas].mean().to_numpy()
    media_total = limpo[continuas].mean().to_numpy()
    desvio_treino = X_train[continuas].std(ddof=0).to_numpy()

    assert np.allclose(scaler.mean_, media_treino)
    assert np.allclose(scaler.scale_, desvio_treino)
    assert not np.allclose(scaler.mean_, media_total, rtol=0, atol=1e-6)
