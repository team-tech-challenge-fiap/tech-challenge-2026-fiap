"""Testes do KNN (T16)."""

import pytest
from sklearn.neighbors import KNeighborsClassifier

from cardio.data.preprocessing import load_splits
from cardio.models.estimators import knn


@pytest.fixture(scope="module")
def treino():
    X_train, _, _, y_train, *_ = load_splits("A", save_ids=False)
    return X_train.head(600), y_train.head(600)


def test_build_model_usa_knn_com_features_escalonadas(treino) -> None:
    X, _ = treino
    pipeline = knn.build_model(X)

    assert isinstance(pipeline.named_steps["model"], KNeighborsClassifier)
    assert pipeline.named_steps["model"].n_neighbors > 5  # k bem maior que o padrão (ruído do dataset)
    assert pipeline.named_steps["prep"].transformers[0][1].__class__.__name__ == "StandardScaler"


def test_param_grid_da_issue() -> None:
    assert knn.PARAM_GRID == {
        "model__n_neighbors": [15, 31, 51, 101],
        "model__weights": ["uniform", "distance"],
    }


def test_knn_treina_e_preve_probabilidades(treino) -> None:
    X, y = treino
    pipeline = knn.build_model(X).fit(X, y)

    proba = pipeline.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
