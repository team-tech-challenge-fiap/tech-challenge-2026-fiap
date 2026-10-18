"""Predição usando um modelo salvo."""

from pathlib import Path

import joblib
import pandas as pd

from ..features.build import build_features


def predict(model_path: Path, data: pd.DataFrame):
    """Carrega o modelo e retorna as classes previstas."""
    model = joblib.load(model_path)
    features = data.drop(columns=["id"], errors="ignore")
    return model.predict(build_features(features))
