"""Configurações do extra de imagens de ECG (``cardio_visual``).

Reaproveita do ``cardio.config`` a semente e as pastas de resultados, para os dois projetos seguirem as mesmas
regras (``SEED = 42`` e split 70/15/15). As constantes próprias do extra ficam aqui, sem mexer na config do
projeto tabular.
"""

from cardio.config import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SEED, TEST_SIZE, VAL_SIZE

__all__ = [
    "CNN_BATCH_SIZE", "CNN_RESULTS_PATH", "CNN_SPLIT_PATH", "FIGURES_DIR", "IMAGE_EXTENSIONS", "IMAGE_SIZE",
    "IMAGES_DIR", "KAGGLE_DATASET", "KAGGLE_URL", "KAGGLE_ZIP_PATH", "MODELS_DIR", "PROJECT_ROOT", "RESULTS_DIR",
    "SEED", "TEST_SIZE", "VAL_SIZE",
]

# --- Dataset (Kaggle) ---
# As imagens não vão para o Git: baixar o zip do Kaggle (ver README) e rodar ``python -m cardio_visual.data.preparar``.
KAGGLE_DATASET = "jayaprakashpondy/ecgimages"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}/data"
KAGGLE_ZIP_PATH = PROJECT_ROOT / "data" / "archive.zip"  # nome padrão do download pelo site do Kaggle
IMAGES_DIR = PROJECT_ROOT / "data" / "images"  # uma pasta por classe: data/images/<classe>/*.jpg
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --- Modelagem ---
IMAGE_SIZE = 224  # lado do quadrado, em pixels, que entra na rede
CNN_BATCH_SIZE = 32
CNN_SPLIT_PATH = RESULTS_DIR / "cnn_split.csv"
CNN_RESULTS_PATH = RESULTS_DIR / "cnn_val_results.csv"
