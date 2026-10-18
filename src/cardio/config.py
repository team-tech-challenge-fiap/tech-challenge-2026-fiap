"""Configurações centrais do projeto: seed, caminhos e contrato do dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardio_train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_DATASET = DATA_PATH  # nome antigo, mantido por compatibilidade

TARGET = "cardio"
REQUIRED_COLUMNS = (
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", TARGET,
)
