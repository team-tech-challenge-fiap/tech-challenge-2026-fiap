"""Testes das features derivadas e dos conjuntos A e B (T10)."""

import pandas as pd
import pytest

from cardio.data.preprocessing import load_splits
from cardio.features.build import (
    FEATURES_A,
    FEATURES_DERIVADAS,
    add_features,
    categorizar_pressao,
    get_feature_names,
    get_feature_set,
)


def _pacientes(**colunas) -> pd.DataFrame:
    """Um paciente com as colunas de ``load_dataset()``; os argumentos sobrescrevem os campos."""
    base = {"age_years": 50.0, "sex_male": 0, "height": 200, "weight": 80, "ap_hi": 130,
            "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
    return pd.DataFrame({**{k: [v] for k, v in base.items()}, **{k: [v] for k, v in colunas.items()}})


def test_add_features_cria_as_cinco_variaveis_derivadas() -> None:
    resultado = add_features(_pacientes())
    assert resultado.loc[0, "imc"] == 20
    assert resultado.loc[0, "pressao_pulso"] == 50
    assert resultado.loc[0, "pam"] == pytest.approx((130 + 2 * 80) / 3)
    assert resultado.loc[0, "categoria_pressao"] == "Hipertensão 1"
    assert resultado.loc[0, "faixa_imc"] == "Normal"
    assert set(FEATURES_DERIVADAS) <= set(resultado.columns)


def test_add_features_nao_altera_o_dataframe_original() -> None:
    original = _pacientes()
    add_features(original)
    assert "imc" not in original.columns


@pytest.mark.parametrize("ap_hi, ap_lo, esperado", [
    (110, 70, "Normal"),
    (125, 75, "Elevada"),
    (135, 75, "Hipertensão 1"),
    (115, 85, "Hipertensão 1"),
    (145, 70, "Hipertensão 2"),
    (125, 95, "Hipertensão 2"),
])
def test_categorizar_pressao_segue_a_diretriz(ap_hi: int, ap_lo: int, esperado: str) -> None:
    assert categorizar_pressao(_pacientes(ap_hi=ap_hi, ap_lo=ap_lo)).loc[0, "categoria_pressao"] == esperado


@pytest.mark.parametrize("peso, esperado", [
    (60, "Abaixo do peso"),   # IMC 15
    (74.9, "Normal"),         # IMC 18,7
    (99.9, "Normal"),         # IMC 24,98: ainda abaixo de 25
    (100, "Sobrepeso"),       # IMC 25
    (120, "Obesidade"),       # IMC 30
])
def test_faixa_imc_usa_os_limites_da_oms(peso: float, esperado: str) -> None:
    resultado = add_features(_pacientes(height=200, weight=peso))
    assert resultado.loc[0, "faixa_imc"] == esperado


def test_get_feature_names_a_e_b() -> None:
    assert get_feature_names("A") == FEATURES_A
    assert get_feature_names("B") == FEATURES_A + FEATURES_DERIVADAS
    with pytest.raises(ValueError, match="inválido"):
        get_feature_names("C")


def test_get_feature_set_devolve_so_as_colunas_do_conjunto() -> None:
    dados = _pacientes()
    assert list(get_feature_set(dados, "A").columns) == FEATURES_A
    assert list(get_feature_set(dados, "B").columns) == FEATURES_A + FEATURES_DERIVADAS


def test_load_splits_b_tem_as_cinco_variaveis_novas_e_o_mesmo_split_do_a() -> None:
    X_a, _, _, y_a, *_ = load_splits("A", save_ids=False)
    X_b, X_val_b, X_test_b, y_b, *_ = load_splits("B", save_ids=False)
    assert set(FEATURES_DERIVADAS) <= set(X_b.columns)
    assert set(X_b.columns) - set(X_a.columns) == set(FEATURES_DERIVADAS)
    assert X_b.index.equals(X_a.index) and y_b.index.equals(y_a.index)  # mesmo split
    for X in (X_b, X_val_b, X_test_b):
        assert not X[FEATURES_DERIVADAS].isna().any().any()
