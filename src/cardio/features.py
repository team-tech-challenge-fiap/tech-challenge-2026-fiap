"""Engenharia de atributos."""

import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Converte idade para anos e adiciona o IMC."""
    features = data.copy()
    features["age_years"] = features["age"] / 365.25
    features["bmi"] = features["weight"] / (features["height"] / 100) ** 2
    return features.drop(columns=["age"])
