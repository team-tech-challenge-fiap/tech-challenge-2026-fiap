import pandas as pd

from cardio.features.build import build_features


def test_build_features_creates_age_years_and_bmi() -> None:
    result = build_features(pd.DataFrame({"age": [365.25], "height": [200], "weight": [80]}))
    assert result.loc[0, "age_years"] == 1
    assert result.loc[0, "bmi"] == 20
