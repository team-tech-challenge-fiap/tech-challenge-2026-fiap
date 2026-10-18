"""Treinamento do modelo baseline."""

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_DATASET
from ..data.loader import load_dataset, split_features_target
from ..features.build import build_features
from ..data.preprocessing import clean_data
from .evaluate import evaluate


def train_model(features, target):
    """Treina uma regressão logística com divisão estratificada."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(x_train, y_train)
    return model, x_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo Cardio.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    data = clean_data(load_dataset(args.dataset))
    raw_features, target = split_features_target(data)
    model, x_test, y_test = train_model(build_features(raw_features), target)
    metrics = evaluate(model, x_test, y_test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "cardio_model.joblib")
    print("\n".join(f"{name}: {value:.4f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
