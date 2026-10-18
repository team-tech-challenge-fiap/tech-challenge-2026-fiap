"""Limpeza e preparação dos dados."""

import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas sem valores e mantém uma cópia independente."""
    return data.dropna().copy()


def remove_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove as colunas informadas sem modificar o DataFrame original."""
    return data.drop(columns=columns).copy()
