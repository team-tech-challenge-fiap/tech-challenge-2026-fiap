"""Testes da estrutura do extra de imagens (CNN-01)."""

import cardio_visual  # noqa: F401  (o pacote importa mesmo sem as imagens e sem o PyTorch)
from cardio import config as config_tabular
from cardio_visual import config


def test_constantes_de_imagem() -> None:
    assert config.IMAGES_DIR == config_tabular.PROJECT_ROOT / "data" / "images"
    assert config.KAGGLE_ZIP_PATH.parent == config_tabular.PROJECT_ROOT / "data"
    assert config.IMAGE_SIZE > 0 and config.CNN_BATCH_SIZE > 0
    assert ".jpg" in config.IMAGE_EXTENSIONS and ".png" in config.IMAGE_EXTENSIONS
    assert config.CNN_SPLIT_PATH.parent == config_tabular.RESULTS_DIR
    assert config.CNN_RESULTS_PATH.parent == config_tabular.RESULTS_DIR
    assert config.KAGGLE_DATASET in config.KAGGLE_URL


def test_usa_a_mesma_semente_e_o_mesmo_split_do_projeto_tabular() -> None:
    assert config.SEED == config_tabular.SEED
    assert (config.VAL_SIZE, config.TEST_SIZE) == (config_tabular.VAL_SIZE, config_tabular.TEST_SIZE)


def test_o_extra_nao_altera_a_config_do_projeto_tabular() -> None:
    assert not hasattr(config_tabular, "IMAGES_DIR") and not hasattr(config_tabular, "CNN_RESULTS_PATH")
