
"""Predição usando um modelo salvo."""

from pathlib import Path

import joblib
import pandas as pd

from ..data.loader import convert_columns
from ..features.build import get_feature_set


def predict(model_path: Path, data: pd.DataFrame, feature_set: str | None = None):
    """Carrega o modelo e retorna as classes previstas.

    Aceita modelos antigos (pipeline) e o artefato final da T23
    (dicionário contendo modelo, features e threshold).

    data contém as colunas originais do CSV.
    """
    artefato = joblib.load(model_path)

    if isinstance(artefato, dict):
        model = artefato["model"]
        conjunto_salvo = artefato["features"]
        limiar = float(artefato["threshold"])

        if feature_set is not None and feature_set != conjunto_salvo:
            raise ValueError(
                f"Features incompatíveis: modelo usa {conjunto_salvo}, "
                f"mas foi solicitado {feature_set}."
            )

        feature_set = conjunto_salvo

    else:
        # Compatibilidade com modelos antigos.
        model = artefato
        feature_set = feature_set or "A"
        limiar = None

    patients = data.drop(columns=["id", "cardio"], errors="ignore").copy()

    if "age" in patients.columns and "gender" in patients.columns:
        patients = convert_columns(patients)
    elif not {"age_years", "sex_male"}.issubset(patients.columns):
        raise ValueError(
        "Os dados precisam conter age/gender ou age_years/sex_male."
    )

    X = get_feature_set(patients, feature_set)

    if limiar is not None:
        probabilidades = model.predict_proba(X)[:, 1]
        return (probabilidades >= limiar).astype(int)

    return model.predict(X)
