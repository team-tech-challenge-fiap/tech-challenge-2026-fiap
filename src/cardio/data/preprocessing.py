"""Limpeza e preparação dos dados."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import (
    CV_FOLDS,
    DATA_PATH,
    PLAUSIBLE_RANGES,
    PRESSURE_RANGES,
    SEED,
    SPLIT_IDS_PATH,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from ..features.build import get_feature_set
from .loader import load_dataset

Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]


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
        df: DataFrame como retornado por ``load_dataset()``.
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


def load_splits(feature_set: str = "A", save_ids: bool = True, verbose: bool = False) -> Splits:
    """Carrega, limpa e divide os dados em treino/validação/teste (70/15/15), estratificado (T09).

    **Todo o time deve usar esta função**, com a mesma ``SEED``, para que os resultados dos
    modelos sejam comparáveis. O teste é usado uma única vez, no final (T23).

    O split é feito em duas etapas com ``train_test_split(stratify=y, random_state=SEED)``:
    primeiro separa o teste (15%), depois divide o restante em treino (70%) e validação (15%).

    Args:
        feature_set: conjunto de features: "A" (variáveis originais limpas) ou "B" (A + as 5
            variáveis derivadas de ``cardio.features.build``).
        save_ids: se True, salva o ``id`` de cada paciente e o conjunto dele em
            ``config.SPLIT_IDS_PATH`` (``results/split_ids.csv``).
        verbose: se True, imprime o log da limpeza e o tamanho de cada conjunto.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``. O índice dos DataFrames/Series
        é o ``id`` do paciente; ``id`` e ``cardio`` nunca aparecem nas features.
    """
    df = clean(load_dataset(DATA_PATH), verbose=verbose).set_index("id")
    X = get_feature_set(df, feature_set)
    y = df[TARGET]

    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_resto, random_state=SEED
    )

    if save_ids:
        SPLIT_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ids = pd.concat([
            pd.DataFrame({"id": y_train.index, "split": "train"}),
            pd.DataFrame({"id": y_val.index, "split": "val"}),
            pd.DataFrame({"id": y_test.index, "split": "test"}),
        ]).sort_values("id")
        ids.to_csv(SPLIT_IDS_PATH, index=False)

    if verbose:
        for nome, alvo in (("Treino", y_train), ("Validação", y_val), ("Teste", y_test)):
            print(f"{nome}: {_inteiro(len(alvo))} pacientes ({_pct(len(alvo) / len(y))}), "
                  f"{_pct(alvo.mean())} com doença")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_cv(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """Validação cruzada estratificada para usar **dentro do treino** (ex.: busca de hiperparâmetros).

    Args:
        n_splits: número de partições (padrão: ``config.CV_FOLDS`` = 5).

    Returns:
        ``StratifiedKFold`` com ``shuffle=True`` e ``random_state=SEED``.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _bmi(df: pd.DataFrame) -> pd.Series:
    """IMC = peso (kg) / altura (m)²."""
    return df["weight"] / (df["height"] / 100) ** 2


def _inteiro(valor: int) -> str:
    """68509 -> '68.509'."""
    return f"{valor:,}".replace(",", ".")


def _pct(valor: float) -> str:
    """0.0213 -> '2,13%'."""
    return f"{valor:.2%}".replace(".", ",")
