import pandas as pd
import pytest

from cardio.data_loader import load_dataset, split_features_target


def test_split_features_target_removes_id_and_target() -> None:
    data = pd.DataFrame({"id": [1], "age": [100], "cardio": [0]})
    features, target = split_features_target(data)
    assert list(features.columns) == ["age"]
    assert target.tolist() == [0]


def test_load_dataset_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "cardio_train.csv"
    path.write_text("id;cardio\n1;0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Colunas necessárias"):
        load_dataset(path)
