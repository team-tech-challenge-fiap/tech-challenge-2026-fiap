"""Configurações centrais do projeto: seed, caminhos e contrato do dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardio_train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
VAL_RESULTS_PATH = RESULTS_DIR / "val_results.csv"

DEFAULT_DATASET = DATA_PATH  # nome antigo, mantido por compatibilidade

TARGET = "cardio"
REQUIRED_COLUMNS = (
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", TARGET,
)

# --- Faixas plausíveis (definidas na T04, aplicadas na limpeza da T08) ---
# Valores fora dessas faixas são considerados erro de registro.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "height": (130, 210),  # altura em cm
    "weight": (35, 200),   # peso em kg
    "imc": (15, 60),       # IMC em kg/m²
}

# --- Faixas de IMC da OMS (usadas na EDA e disponíveis para a T10) ---
IMC_BINS = [0, 18.5, 25, 30, 35, 40, float("inf")]
IMC_LABELS = ["Abaixo do peso", "Normal", "Sobrepeso", "Obesidade I", "Obesidade II", "Obesidade III"]

# --- Faixas plausíveis de pressão arterial (definidas na T05, aplicadas na limpeza da T08) ---
PRESSURE_RANGES: dict[str, tuple[float, float]] = {
    "ap_hi": (70, 250),  # sistólica em mmHg
    "ap_lo": (40, 200),  # diastólica em mmHg
}

# --- Split estratificado 70/15/15 (T09) ---
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5
SPLIT_IDS_PATH = RESULTS_DIR / "split_ids.csv"
