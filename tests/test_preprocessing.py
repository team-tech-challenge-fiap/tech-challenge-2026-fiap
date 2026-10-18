"""Testes da limpeza dos dados (T08)."""

import pandas as pd

from cardio.config import DATA_PATH, PLAUSIBLE_RANGES, PRESSURE_RANGES
from cardio.data.loader import load_dataset
from cardio.data.preprocessing import clean


def _paciente(**valores) -> dict:
    """Paciente válido; os argumentos sobrescrevem os campos desejados."""
    base = {"id": 0, "age": 18393, "gender": 2, "height": 168, "weight": 62.0, "ap_hi": 110,
            "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1, "cardio": 0}
    return {**base, **valores}


def test_clean_remove_cada_tipo_de_erro() -> None:
    df = pd.DataFrame([
        _paciente(id=1),                          # válido
        _paciente(id=2),                          # duplicado do id=1 (só o id muda)
        _paciente(id=3, ap_hi=16020),             # sistólica absurda
        _paciente(id=4, ap_lo=-70),               # diastólica negativa
        _paciente(id=5, ap_hi=80, ap_lo=120),     # diastólica > sistólica
        _paciente(id=6, height=55),               # altura impossível
        _paciente(id=7, weight=10),               # peso impossível
        _paciente(id=8, height=130, weight=110),  # IMC > 60 (altura e peso válidos separados)
        _paciente(id=9, cardio=1, age=20000),     # válido
    ])
    resultado = clean(df, verbose=False)
    assert resultado["id"].tolist() == [1, 9]
    assert resultado.attrs["cleaning_log"]["linhas_removidas"].tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_clean_nao_altera_o_dataframe_original() -> None:
    df = pd.DataFrame([_paciente(id=1), _paciente(id=2, height=55)])
    clean(df, verbose=False)
    assert len(df) == 2


def test_clean_no_dataset_real_deixa_tudo_dentro_das_faixas() -> None:
    bruto = load_dataset(DATA_PATH)
    limpo = clean(bruto, verbose=False)

    for coluna, (minimo, maximo) in {**PRESSURE_RANGES, **PLAUSIBLE_RANGES}.items():
        valores = limpo["weight"] / (limpo["height"] / 100) ** 2 if coluna == "bmi" else limpo[coluna]
        assert valores.between(minimo, maximo).all(), f"{coluna} fora da faixa"
    assert (limpo["ap_lo"] < limpo["ap_hi"]).all()
    assert not limpo.drop(columns="id").duplicated().any()

    # O log precisa bater com o total removido
    assert limpo.attrs["cleaning_log"]["linhas_removidas"].sum() == len(bruto) - len(limpo)

    # A limpeza não pode desbalancear as classes (diferença < 1 ponto percentual)
    assert abs(bruto["cardio"].mean() - limpo["cardio"].mean()) < 0.01
