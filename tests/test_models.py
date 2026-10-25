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


@pytest.mark.parametrize("conjunto", ["A", "B"])
@pytest.mark.parametrize("nome", MODELOS)
def test_modelo_segue_o_contrato(nome: str, conjunto: str) -> None:
    modulo = import_module(f"cardio.models.estimators.{nome}")
    X_train, _, _, y_train, *_ = load_splits(conjunto, save_ids=False)
    X_train, y_train = X_train.head(300), y_train.head(300)

    pipeline = modulo.build_model(X_train)

    assert isinstance(pipeline, Pipeline)
    assert [passo for passo, _ in pipeline.steps] == ["prep", "model"]  # PARAM_GRID usa "model__"
    assert all(chave.startswith("model__") for chave in modulo.PARAM_GRID)
    assert pipeline.fit(X_train, y_train).predict(X_train).shape == (300,)
