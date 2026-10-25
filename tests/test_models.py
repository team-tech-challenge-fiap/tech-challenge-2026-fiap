"""Testes do contrato dos modelos (T13): cada arquivo de ``estimators`` expõe build_model() e PARAM_GRID."""

import pkgutil
from importlib import import_module

import pytest
from sklearn.pipeline import Pipeline

from cardio.data.preprocessing import load_splits
from cardio.models import estimators

MODELOS = [modulo.name for modulo in pkgutil.iter_modules(estimators.__path__)]


def test_ha_modelos_para_testar() -> None:
    assert {"example", "random_forest"} <= set(MODELOS)


@pytest.mark.parametrize("nome", MODELOS)
def test_modelo_segue_o_contrato(nome: str) -> None:
    modulo = import_module(f"cardio.models.estimators.{nome}")
    X_train = load_splits("A", save_ids=False)[0].head(200)

    pipeline = modulo.build_model(X_train)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[-1][0] == "model"  # os PARAM_GRID usam o prefixo "model__"
    assert all(chave.startswith("model__") for chave in modulo.PARAM_GRID)
