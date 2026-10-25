"""T13 · Modelo de exemplo (baseline de ~50% que os demais precisam superar)."""

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ...data.preprocessing import build_preprocessor

PARAM_GRID = {}


def build_model(X_train):
    """Constrói o pipeline do modelo baseline."""
    model = DummyClassifier(strategy="most_frequent")

    return Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", model),
    ])
