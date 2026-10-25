"""T17 · Random Forest: contrato da T13 (``build_model()`` + ``PARAM_GRID``)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ...config import SEED

PARAM_GRID = {
    "model__max_depth": [6, 10, 14, None],
    "model__min_samples_leaf": [1, 20, 50],
    "model__max_features": ["sqrt", 0.5],
}


def build_model(X_train):
    """Constrói o pipeline do Random Forest."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=SEED,
    )

    return Pipeline([
        # TODO (T11): ("prep", build_preprocessor(X_train, scale=False)),
        ("model", model),
    ])
