"""Avaliação do modelo."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate(model, features, target) -> dict[str, float]:
    """Calcula métricas básicas de classificação binária."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }
