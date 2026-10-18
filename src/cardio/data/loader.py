"""Carregamento e validação do dataset."""

from pathlib import Path

import pandas as pd

from ..config import REQUIRED_COLUMNS, TARGET

def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV e valida as colunas mínimas esperadas."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    data = pd.read_csv(path, sep=";")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Colunas necessárias não encontradas: {', '.join(missing)}")
    return data


def convert_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Converte ``age`` e ``gender`` para o formato usado na modelagem (T02).
    - ``age_years = age / 365.25``: a idade vem em **dias** no CSV;
    - ``sex_male = (gender == 2)``: ``gender`` usa 1 = mulher e 2 = homem. ``sex_male`` vira uma binária explícita: 1 = homem, 0 = mulher.
    - ``age`` e ``gender`` são removidos. """

    converted = data.copy()
    converted["age_years"] = converted["age"] / 365.25
    converted["sex_male"] = (converted["gender"] == 2).astype(int)
    return converted.drop(columns=["age", "gender"])


def load_dataset(path: Path) -> pd.DataFrame:
    """Lê o CSV (``load_raw``) e aplica as conversões de idade e sexo (``convert_columns``). """
    return convert_columns(load_raw(path))


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos e alvo, removendo o identificador."""
    return data.drop(columns=["id", TARGET]), data[TARGET].copy()
