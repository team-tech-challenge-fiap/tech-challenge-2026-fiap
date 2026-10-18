"""Configurações e contrato do dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "cardio_train.csv"
TARGET = "cardio"
REQUIRED_COLUMNS = (
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", TARGET,
)
