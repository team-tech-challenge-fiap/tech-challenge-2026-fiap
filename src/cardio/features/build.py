"""Engenharia de atributos e conjuntos de features A e B (T10).

As variáveis derivadas são funções diretas das originais (nunca usam o alvo), então podem ser
calculadas antes do split sem vazamento de dados. As colunas de entrada são as que saem de
``cardio.data.loader.load_dataset`` (``age_years`` e ``sex_male`` no lugar de ``age`` e ``gender``).
"""

import numpy as np
import pandas as pd

from ..config import BMI_BINS

# --- Conjunto A: variáveis originais limpas ---
FEATURES_A = [
    "age_years", "sex_male", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]

# --- Conjunto B: A + variáveis derivadas ---
FEATURES_DERIVADAS = ["imc", "pressao_pulso", "pam", "categoria_pressao", "faixa_imc"]
FEATURES_B = FEATURES_A + FEATURES_DERIVADAS

# Mesmas faixas de IMC da OMS do config, agrupando tudo a partir de 30 como "Obesidade".
FAIXA_IMC_BINS = BMI_BINS[:4] + [float("inf")]
FAIXA_IMC_LABELS = ["Abaixo do peso", "Normal", "Sobrepeso", "Obesidade"]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as variáveis derivadas (conjunto B) a uma cópia do DataFrame.

    - ``imc`` = peso / (altura em m)²: resume peso e altura num único indicador de adiposidade.
    - ``pressao_pulso`` = ``ap_hi`` - ``ap_lo``: reflete a rigidez das artérias.
    - ``pam`` = (``ap_hi`` + 2 × ``ap_lo``) / 3: pressão arterial média, a pressão de perfusão dos órgãos.
    - ``categoria_pressao``: normal, elevada, hipertensão 1 ou 2 (ver ``categorizar_pressao``).
    - ``faixa_imc``: abaixo do peso, normal, sobrepeso ou obesidade (faixas da OMS).
    """
    features = data.copy()
    features["imc"] = features["weight"] / (features["height"] / 100) ** 2
    features["pressao_pulso"] = features["ap_hi"] - features["ap_lo"]
    features["pam"] = (features["ap_hi"] + 2 * features["ap_lo"]) / 3
    features = categorizar_pressao(features)
    features["faixa_imc"] = pd.cut(
        features["imc"], bins=FAIXA_IMC_BINS, labels=FAIXA_IMC_LABELS, right=False
    ).astype(str)
    return features


def categorizar_pressao(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona ``categoria_pressao`` seguindo a diretriz ACC/AHA 2017 (mesmas faixas da T05).

    - Normal: sistólica < 120 e diastólica < 80;
    - Elevada: sistólica 120–129 e diastólica < 80;
    - Hipertensão 1: sistólica 130–139 ou diastólica 80–89;
    - Hipertensão 2: sistólica ≥ 140 ou diastólica ≥ 90.

    Quando sistólica e diastólica caem em categorias diferentes, vale a mais alta. É um atributo
    para o modelo, não um diagnóstico clínico.
    """
    result = data.copy()
    condicoes = [
        (result["ap_hi"] >= 140) | (result["ap_lo"] >= 90),
        (result["ap_hi"] >= 130) | (result["ap_lo"] >= 80),
        (result["ap_hi"] >= 120) & (result["ap_lo"] < 80),
    ]
    categorias = ["Hipertensão 2", "Hipertensão 1", "Elevada"]
    result["categoria_pressao"] = np.select(condicoes, categorias, default="Normal")
    return result


def get_feature_names(name: str) -> list[str]:
    """Nomes das colunas do conjunto de features: ``"A"`` (originais) ou ``"B"`` (A + derivadas)."""
    if name == "A":
        return FEATURES_A.copy()
    if name == "B":
        return FEATURES_B.copy()
    raise ValueError(f"Conjunto de features inválido: {name!r}. Use 'A' ou 'B'.")


def get_feature_set(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """Devolve só as colunas do conjunto ``name``, calculando as derivadas quando for o "B"."""
    columns = get_feature_names(name)
    if name == "B":
        data = add_features(data)
    return data[columns]
