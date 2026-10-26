"""Testes do ajuste do limiar de decisão (T22)."""

import numpy as np
import pytest

from cardio.models.evaluate import choose_threshold, impact_per_1000, threshold_curve

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
PROBA = np.array([0.05, 0.20, 0.35, 0.60, 0.30, 0.40, 0.55, 0.70, 0.80, 0.90])


def test_curva_tem_um_limiar_por_linha_e_recall_cai_com_o_limiar() -> None:
    curva = threshold_curve(Y, PROBA)
    assert curva["limiar"].iloc[0] == pytest.approx(0.10)
    assert curva["limiar"].iloc[-1] == pytest.approx(0.90)
    assert curva["recall"].is_monotonic_decreasing


def test_escolhe_o_maior_limiar_com_recall_minimo() -> None:
    # recall >= 0,8 exige sinalizar 5 dos 6 doentes: o maior limiar que faz isso é 0,40
    assert choose_threshold(Y, PROBA, min_recall=0.8) == pytest.approx(0.40)
    # recall 100% exige pegar o doente com 0,30
    assert choose_threshold(Y, PROBA, min_recall=1.0) == pytest.approx(0.30)


def test_erro_quando_nenhum_limiar_atinge_o_recall() -> None:
    with pytest.raises(ValueError):
        choose_threshold(Y, PROBA, min_recall=1.0, thresholds=[0.5, 0.6])


def test_impacto_por_1000_pacientes() -> None:
    impacto = impact_per_1000(Y, PROBA, 0.5)
    # Com 0,5: 2 doentes perdidos (0,30 e 0,40) e 1 saudável encaminhado (0,60), em 10 pacientes
    assert impacto["doentes_nao_sinalizados_por_1000"] == pytest.approx(200)
    assert impacto["saudaveis_encaminhados_por_1000"] == pytest.approx(100)
    assert impacto["especificidade"] == pytest.approx(0.75)
