"""Limpeza e preparação dos dados."""

import pandas as pd

from ..config import PLAUSIBLE_RANGES, PRESSURE_RANGES, TARGET


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas sem valores e mantém uma cópia independente."""
    return data.dropna().copy()


def remove_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove as colunas informadas sem modificar o DataFrame original."""
    return data.drop(columns=columns).copy()


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove duplicados e registros fisiologicamente impossíveis (T08).

    As regras são aplicadas em sequência, e cada uma conta só as linhas que ela
    removeu (as já removidas por uma regra anterior não entram de novo na conta):

    1. Duplicados: linhas idênticas ignorando o ``id``.
    2. Pressão sistólica (``ap_hi``) fora de ``config.PRESSURE_RANGES`` (T05).
    3. Pressão diastólica (``ap_lo``) fora de ``config.PRESSURE_RANGES`` (T05).
    4. Diastólica maior ou igual à sistólica: impossível fisiologicamente (T05).
    5. Altura fora de ``config.PLAUSIBLE_RANGES`` (T04).
    6. Peso fora de ``config.PLAUSIBLE_RANGES`` (T04).
    7. IMC fora de ``config.PLAUSIBLE_RANGES`` (T04): pega combinações altura/peso inválidas.

    Os registros com erro são removidos, e não corrigidos, seguindo a recomendação da T05.

    Args:
        df: DataFrame bruto, como retornado por ``load_dataset()``.
        verbose: se True, imprime o log de remoções e a taxa de doença antes e depois.

    Returns:
        Cópia limpa do DataFrame, com o índice reiniciado e as mesmas colunas da entrada.
        O log de remoções fica em ``resultado.attrs["cleaning_log"]`` (DataFrame).
    """
    data = df.copy()
    total_inicial = len(data)
    taxa_inicial = data[TARGET].mean()

    regras = [
        ("Duplicados (ignorando o id)",
         lambda d: ~d.drop(columns=["id"], errors="ignore").duplicated()),
        (f"Sistólica fora de {PRESSURE_RANGES['ap_hi'][0]}–{PRESSURE_RANGES['ap_hi'][1]} mmHg",
         lambda d: d["ap_hi"].between(*PRESSURE_RANGES["ap_hi"])),
        (f"Diastólica fora de {PRESSURE_RANGES['ap_lo'][0]}–{PRESSURE_RANGES['ap_lo'][1]} mmHg",
         lambda d: d["ap_lo"].between(*PRESSURE_RANGES["ap_lo"])),
        ("Diastólica >= sistólica",
         lambda d: d["ap_lo"] < d["ap_hi"]),
        (f"Altura fora de {PLAUSIBLE_RANGES['height'][0]}–{PLAUSIBLE_RANGES['height'][1]} cm",
         lambda d: d["height"].between(*PLAUSIBLE_RANGES["height"])),
        (f"Peso fora de {PLAUSIBLE_RANGES['weight'][0]}–{PLAUSIBLE_RANGES['weight'][1]} kg",
         lambda d: d["weight"].between(*PLAUSIBLE_RANGES["weight"])),
        (f"IMC fora de {PLAUSIBLE_RANGES['bmi'][0]}–{PLAUSIBLE_RANGES['bmi'][1]} kg/m²",
         lambda d: _bmi(d).between(*PLAUSIBLE_RANGES["bmi"])),
    ]

    log = []
    for motivo, manter in regras:
        mascara = manter(data)
        log.append({"regra": motivo, "linhas_removidas": int((~mascara).sum())})
        data = data[mascara]

    data = data.reset_index(drop=True)
    log_df = pd.DataFrame(log)
    data.attrs["cleaning_log"] = log_df

    if verbose:
        removidas = total_inicial - len(data)
        print(f"{_inteiro(total_inicial)} → {_inteiro(len(data))} linhas "
              f"({_inteiro(removidas)} removidas, {_pct(removidas / total_inicial)})")
        for linha in log:
            print(f"  - {linha['regra']}: {_inteiro(linha['linhas_removidas'])}")
        print(f"Taxa de doença: {_pct(taxa_inicial)} → {_pct(data[TARGET].mean())}")

    return data


def _bmi(df: pd.DataFrame) -> pd.Series:
    """IMC = peso (kg) / altura (m)²."""
    return df["weight"] / (df["height"] / 100) ** 2


def _inteiro(valor: int) -> str:
    """68509 -> '68.509'."""
    return f"{valor:,}".replace(",", ".")


def _pct(valor: float) -> str:
    """0.0213 -> '2,13%'."""
    return f"{valor:.2%}".replace(".", ",")
