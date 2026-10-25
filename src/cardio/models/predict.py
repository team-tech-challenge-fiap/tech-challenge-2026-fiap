"""Predição usando um modelo salvo."""

from pathlib import Path

import joblib
import pandas as pd

from ..data.loader import convert_columns
from ..features.build import get_feature_set


def predict(model_path: Path, data: pd.DataFrame, feature_set: str = "A"):
    """Carrega o modelo e retorna as classes previstas.

    ``data`` tem as colunas originais do CSV (``age`` em dias e ``gender``); ``feature_set`` deve
    ser o mesmo conjunto com que o modelo foi treinado.
    """
    model = joblib.load(model_path)
    patients = convert_columns(data.drop(columns=["id"], errors="ignore"))
    return model.predict(get_feature_set(patients, feature_set))
