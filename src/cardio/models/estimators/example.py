"""T13 · Modelo de exemplo (baseline de ~50% que os demais precisam superar)."""

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

PARAM_GRID = {}


def build_model(X_train):
    """Constrói o pipeline do modelo baseline."""
    model = DummyClassifier(strategy="most_frequent")

    return Pipeline([
        # TODO (T11): ("prep", build_preprocessor(X_train)),
        ("model", model),
    ])
