import pandas as pd
import pytest

from cardio.config import DATA_PATH
from cardio.data.loader import (
    convert_columns,
    load_dataset,
    load_raw,
    split_features_target,
)
from cardio.data.preprocessing import load_splits


def test_split_features_target_removes_id_and_target() -> None:
    data = pd.DataFrame({"id": [1], "age": [100], "cardio": [0]})
    features, target = split_features_target(data)
    assert list(features.columns) == ["age"]
    assert target.tolist() == [0]


def test_load_raw_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "cardio_train.csv"
    path.write_text("id;cardio\n1;0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Colunas necessárias"):
        load_raw(path)


def test_convert_columns_creates_age_years_and_sex_male() -> None:
    data = pd.DataFrame({"id": [1, 2], "age": [365.25, 7305], "gender": [1, 2], "cardio": [0, 1]})
    result = convert_columns(data)
    assert result["age_years"].tolist() == [1, 20]
    assert result["sex_male"].tolist() == [0, 1]
    assert list(result.columns) == ["id", "cardio", "age_years", "sex_male"]
    assert "age" in data.columns and "gender" in data.columns  # entrada não é alterada


def test_load_dataset_real_converte_colunas_e_mantem_id() -> None:
    bruto = load_raw(DATA_PATH)
    dataset = load_dataset(DATA_PATH)
    assert len(dataset) == len(bruto) == 70_000
    assert {"id", "cardio", "age_years", "sex_male"} <= set(dataset.columns)
    assert not {"age", "gender"} & set(dataset.columns)
    assert dataset["age_years"].between(29, 66).all()
    assert set(dataset["sex_male"].unique()) == {0, 1}


def test_load_splits_entrega_sex_male_e_age_years_sem_id_age_gender() -> None:
    X_train, X_val, X_test, *_ = load_splits("A", save_ids=False)
    for X in (X_train, X_val, X_test):
        assert {"age_years", "sex_male"} <= set(X.columns)
        assert not {"id", "age", "gender", "cardio"} & set(X.columns)
