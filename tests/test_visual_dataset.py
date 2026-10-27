"""Testes da leitura e da verificação de integridade das imagens (CNN-02), com imagens sintéticas."""

import shutil

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from cardio_visual.data.loader import find_duplicates, inspect_images, integrity_report, scan_images  # noqa: E402


def _salvar(caminho, semente, lado=32):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(semente).integers(0, 256, (lado, lado), dtype=np.uint8)
    Image.fromarray(pixels).save(caminho)


@pytest.fixture()
def dataset(tmp_path):
    """Duas classes, grupos Training/Testing, 1 arquivo corrompido e duplicatas entre classes e entre grupos."""
    raiz = tmp_path / "imagens"
    for i in range(3):
        _salvar(raiz / "Training" / "tumor" / f"t{i}.jpg", i)
        _salvar(raiz / "Training" / "sem_tumor" / f"s{i}.png", 10 + i)
    _salvar(raiz / "Testing" / "tumor" / "t_teste.jpg", 100)
    (raiz / "Training" / "tumor" / "quebrada.jpg").write_bytes(b"isto nao e uma imagem")
    shutil.copy(raiz / "Training" / "tumor" / "t0.jpg", raiz / "Training" / "sem_tumor" / "t0_rotulo_errado.jpg")
    shutil.copy(raiz / "Training" / "tumor" / "t1.jpg", raiz / "Testing" / "tumor" / "t1_vazou.jpg")
    (raiz / "Training" / "tumor" / "leiame.txt").write_text("ignorado")
    return raiz


def test_scan_images_le_classe_e_grupo(dataset) -> None:
    imagens = scan_images(dataset)

    assert set(imagens["classe"]) == {"tumor", "sem_tumor"}
    assert set(imagens["grupo_original"]) == {"treino", "teste"}
    assert not imagens["caminho"].str.endswith(".txt").any()  # só extensões de imagem
    assert imagens["caminho"].is_monotonic_increasing  # ordem estável (reprodutível)
    teste = imagens[imagens["grupo_original"] == "teste"]
    assert set(teste["classe"]) == {"tumor"}


def test_scan_images_sem_grupo_quando_nao_ha_pasta_de_grupo(tmp_path) -> None:
    for classe in ("a", "b"):
        _salvar(tmp_path / classe / "x.jpg", 1)
    assert set(scan_images(tmp_path)["grupo_original"]) == {""}


def test_scan_images_erros(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_images(tmp_path / "nao_existe")
    with pytest.raises(ValueError, match="Nenhuma imagem"):
        scan_images(tmp_path)
    _salvar(tmp_path / "so_uma" / "x.jpg", 1)
    with pytest.raises(ValueError, match="2 classes"):
        scan_images(tmp_path)


def test_inspect_images_acha_a_corrompida_e_mede_as_validas(dataset) -> None:
    inspecionadas = inspect_images(scan_images(dataset))

    corrompidas = inspecionadas[inspecionadas["erro"].notna()]
    assert corrompidas["caminho"].str.endswith("quebrada.jpg").all() and len(corrompidas) == 1
    validas = inspecionadas[inspecionadas["erro"].isna()]
    assert (validas["largura"] == 32).all() and (validas["altura"] == 32).all()
    assert validas["md5"].notna().all() and validas["hash_perceptual"].str.len().eq(64).all()


def test_find_duplicates_agrupa_as_copias(dataset) -> None:
    duplicadas = find_duplicates(inspect_images(scan_images(dataset)))

    assert len(duplicadas) == 4  # 2 pares: t0 e a cópia com rótulo errado, t1 e a cópia que vazou
    assert duplicadas["id_duplicata"].nunique() == 2


def test_integrity_report_resume_o_que_importa(dataset) -> None:
    relatorio = integrity_report(inspect_images(scan_images(dataset)))

    assert relatorio["total"] == 3 + 3 + 1 + 1 + 2  # 10 imagens, 1 delas corrompida
    assert len(relatorio["corrompidas"]) == 1
    assert relatorio["duplicatas_exatas"] == 2  # uma cópia extra em cada par
    assert relatorio["duplicatas_entre_classes"] == 1  # t0 com rótulo diferente
    assert relatorio["duplicatas_entre_grupos"] == 1  # t1 em treino e teste
    assert relatorio["por_classe"]["tumor"] == 6
