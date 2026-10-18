import pandas as pd

from cardio.features.build import build_features


def test_build_features_creates_bmi() -> None:
    result = build_features(pd.DataFrame({"height": [200], "weight": [80]}))
    assert result.loc[0, "bmi"] == 20
