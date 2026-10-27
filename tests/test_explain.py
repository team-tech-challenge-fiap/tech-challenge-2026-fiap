"""Testes da interpretação com SHAP (T25)."""

import numpy as np
import pytest

from cardio.data.preprocessing import load_splits
from cardio.models.estimators.xgb import build_model
from cardio.models.explain import probabilidade, shap_por_faixa, shap_values


@pytest.fixture(scope="module")
def explicacao_e_modelo():
    X_train, X_val, _, y_train, _, _ = load_splits("A", save_ids=False)
    modelo = build_model(X_train).fit(X_train.head(3000), y_train.head(3000))
    explicacao, X_amostra = shap_values(modelo, X_val, sample_size=200)
    return explicacao, X_amostra, modelo


def test_shap_soma_reproduz_a_previsao(explicacao_e_modelo) -> None:
    explicacao, X_amostra, modelo = explicacao_e_modelo
    soma = explicacao.base_values + explicacao.values.sum(axis=1)
    assert np.allclose(probabilidade(soma), modelo.predict_proba(X_amostra)[:, 1], atol=1e-5)


def test_nomes_clinicos_e_tamanho_da_amostra(explicacao_e_modelo) -> None:
    explicacao, X_amostra, _ = explicacao_e_modelo
    assert explicacao.values.shape == (200, 11)
    assert len(X_amostra) == 200
    assert "Pressão sistólica (mmHg)" in explicacao.feature_names


def test_shap_por_faixa_conta_todos_os_pacientes(explicacao_e_modelo) -> None:
    explicacao, _, _ = explicacao_e_modelo
    tabela = shap_por_faixa(explicacao, "Idade (anos)", [0, 50, 70])
    assert tabela["pacientes"].sum() == 200
