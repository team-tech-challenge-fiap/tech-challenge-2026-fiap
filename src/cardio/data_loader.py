"""Carregamento e validação do dataset."""

from pathlib import Path

import pandas as pd

from .config import REQUIRED_COLUMNS, TARGET

def load_dataset(path: Path) -> pd.DataFrame:
    """Lê o CSV e valida as colunas mínimas esperadas."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    data = pd.read_csv(path, sep=";")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Colunas necessárias não encontradas: {', '.join(missing)}")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos e alvo, removendo o identificador."""
    return data.drop(columns=["id", TARGET]), data[TARGET].copy()
