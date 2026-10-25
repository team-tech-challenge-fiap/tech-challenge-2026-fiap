import pandas as pd

from cardio.features import add_features


def test_add_features_creates_derived_features() -> None:
    result = add_features(pd.DataFrame({
        "age": [50], "gender": [1], "height": [200], "weight": [80],
        "ap_hi": [130], "ap_lo": [80], "cholesterol": [1], "gluc": [1],
        "smoke": [0], "alco": [0], "active": [1],
    }))
    assert result.loc[0, "imc"] == 20
    assert result.loc[0, "pressao_pulso"] == 50
