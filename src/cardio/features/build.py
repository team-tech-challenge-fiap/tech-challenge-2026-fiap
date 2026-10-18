"""Engenharia de atributos."""

import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona o IMC."""
    features = data.copy()
    features["bmi"] = features["weight"] / (features["height"] / 100) ** 2
    return features
