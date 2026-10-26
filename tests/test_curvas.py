"""Testes das curvas ROC, Precision-Recall e calibração (T21)."""

import numpy as np
import pandas as pd
import pytest

from cardio.models.evaluate import available_models, make_curve_figures, validation_probabilities
from cardio.visualization.plots import plot_calibration, plot_pr_curves, plot_roc_curves


@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    bom = np.clip(0.5 * y + rng.normal(0.25, 0.2, 400), 0, 1)
    ruim = rng.random(400)
    return {"bom": bom, "ruim": ruim}, y


def test_available_models_ignora_o_example() -> None:
    nomes = available_models()
    assert "example" not in nomes
    assert {"logistic", "random_forest", "xgb"} <= set(nomes)


@pytest.mark.parametrize("funcao", [plot_roc_curves, plot_pr_curves])
def test_curvas_salvam_png(sintetico, tmp_path, funcao) -> None:
    proba, y = sintetico
    caminho = funcao(proba, y, tmp_path / "sub" / "curva.png")
    assert caminho.exists() and caminho.stat().st_size > 0


def test_calibracao_salva_png(sintetico, tmp_path) -> None:
    proba, y = sintetico
    assert plot_calibration(proba["bom"], y, tmp_path / "cal.png", "bom").exists()


def test_validation_probabilities_devolve_probabilidades_validas() -> None:
    proba, y_val = validation_probabilities("A", ["logistic"])
    assert set(proba) == {"logistic"}
    assert len(proba["logistic"]) == len(y_val)
    assert ((proba["logistic"] >= 0) & (proba["logistic"] <= 1)).all()


def test_make_curve_figures_gera_as_quatro_figuras(tmp_path) -> None:
    figuras = make_curve_figures(tmp_path, models=["logistic"])
    assert set(figuras) == {"roc", "pr", "a_vs_b", "calibration"}
    assert all(caminho.exists() for caminho in figuras.values())
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "calibration_logistic.png", "pr_A.png", "roc_A.png", "roc_A_vs_B_logistic.png"]
