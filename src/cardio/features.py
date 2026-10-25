"""Engenharia de atributos."""

import pandas as pd
import numpy as np


FEATURES_A = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol",
    "gluc", "smoke", "alco", "active",
]

FEATURES_B = ["imc", "pressao_pulso", "pam", "categoria_pressao", "faixa_imc"]


def add_features(x):
    """Adiciona atributos derivados ao dataframe."""
    df = x.copy()

    df["imc"] = df["weight"] / (df["height"] / 100) ** 2
    df["pressao_pulso"] = df["ap_hi"] - df["ap_lo"]
    df["pam"] = (df["ap_hi"] + 2 * df["ap_lo"]) / 3
        
    df = categorizar_pressao(df)

    df["faixa_imc"] = pd.cut(
        df["imc"],
        bins=[0, 18.5, 24.9, 29.9, float("inf")],
        labels=["Abaixo do Peso", "Normal", "Sobrepeso", "Obesidade"],
        right=False,
    )

    return df


def get_feature(name):
    """Retorna o conjunto de atributos correspondente ao nome."""
    if name == "A":
        return FEATURES_A.copy()
    if name == "B":
        return FEATURES_A + FEATURES_B
    raise ValueError("Conjunto inválido. Use 'A' ou 'B'.")


def categorizar_pressao(x):
    df = x.copy()

    condicoes = [
        (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90),
        (df["ap_hi"] >= 130) | (df["ap_lo"] >= 80),
        (df["ap_hi"] >= 120) & (df["ap_lo"] < 80),
        (df["ap_hi"] < 120) & (df["ap_lo"] < 80)
    ]

    categorias = [
        "Hipertensão 2",
        "Hipertensão 1",
        "Elevada",
        "Normal"
    ]

    df["categoria_pressao"] = np.select(
        condicoes,
        categorias,
        default="Normal"
    )
    return df