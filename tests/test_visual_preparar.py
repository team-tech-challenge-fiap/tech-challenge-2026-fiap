"""Testes da preparação das imagens a partir do zip do Kaggle (CNN-02), com um zip sintético."""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from cardio_visual.data.preparar import class_name, main, prepare_images, reserve_demo  # noqa: E402

INFARTO = "ECG Images of Myocardial Infarction Patients (240x12=2880)"
HISTORICO = "ECG Images of Patient that have History of MI (172x12=2064)"
ANORMAL = "ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
NORMAL = "Normal Person ECG Images (284x12=3408)"


def _jpg(semente: int) -> bytes:
    pixels = np.random.default_rng(semente).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def zip_kaggle(tmp_path):
    """Zip com o formato do Kaggle: train/ e test/ com as mesmas classes, cópias, conflito e arquivos estranhos."""
    caminho = tmp_path / "archive.zip"
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr(f"train/{INFARTO}/train (1).jpg", _jpg(1))
        z.writestr(f"train/{INFARTO}/train (2).jpg", _jpg(1))  # cópia dentro do train
        z.writestr(f"test/{INFARTO}/test (1).jpg", _jpg(1))  # cópia no test (vazamento no zip original)
        z.writestr(f"train/{HISTORICO}/a.jpg", _jpg(2))
        z.writestr(f"train/{ANORMAL}/b.jpg", _jpg(3))
        z.writestr(f"train/{NORMAL}/c.jpg", _jpg(4))
        z.writestr(f"train/{NORMAL}/conflito.jpg", _jpg(5))
        z.writestr(f"train/{ANORMAL}/conflito.jpg", _jpg(5))  # mesma imagem em duas classes: rótulo incerto
        z.writestr("../fora_da_pasta.jpg", _jpg(6))
        z.writestr("raiz_sem_classe.jpg", _jpg(7))
        z.writestr(f"train/{NORMAL}/leiame.txt", "não é imagem")
    return caminho


def test_nomes_curtos_das_classes_do_kaggle() -> None:
    assert class_name(INFARTO) == "infarto"
    assert class_name(HISTORICO) == "historico_infarto"
    assert class_name(ANORMAL) == "batimento_anormal"
    assert class_name(NORMAL) == "normal"
    assert class_name("Outra Pasta (x)") == "outra_pasta_x"


def test_junta_train_e_test_numa_pasta_por_classe_sem_copias(zip_kaggle, tmp_path) -> None:
    destino = tmp_path / "images"
    resumo = prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False).set_index("classe")

    pastas = sorted(p.name for p in destino.iterdir() if p.is_dir())
    assert pastas == ["batimento_anormal", "historico_infarto", "infarto", "normal"]  # sem train/ e test/
    assert len(list((destino / "infarto").iterdir())) == 1  # 3 cópias viram 1
    assert resumo.loc["infarto", "arquivos_no_zip"] == 3 and resumo.loc["infarto", "imagens_unicas"] == 1
    assert resumo.loc["total", "imagens_unicas"] == 4  # infarto, histórico, anormal e normal (o conflito sai)


def test_imagem_com_classes_diferentes_e_descartada(zip_kaggle, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False)
    origem = pd.read_csv(destino / "origem.csv")

    assert not origem["arquivos_originais"].str.contains("conflito").any()
    assert len(list((destino / "normal").iterdir())) == 1 and len(list((destino / "batimento_anormal").iterdir())) == 1


def test_nada_e_gravado_fora_do_destino_e_so_imagens_de_classe_entram(zip_kaggle, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False)

    assert not (tmp_path / "fora_da_pasta.jpg").exists()
    arquivos = [p for p in destino.rglob("*") if p.is_file()]
    assert {p.suffix for p in arquivos} == {".jpg", ".csv"}
    assert all(p.parent != destino or p.name == "origem.csv" for p in arquivos)  # nenhuma imagem solta na raiz


def test_origem_registra_de_onde_veio_cada_imagem(zip_kaggle, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False)
    origem = pd.read_csv(destino / "origem.csv").set_index("classe")

    assert origem.loc["infarto", "copias"] == 3
    originais = origem.loc["infarto", "arquivos_originais"]
    assert "test/" in originais and "train/" in originais
    assert (destino / origem.loc["infarto", "arquivo"]).exists()


def test_nao_refaz_se_ja_estiver_preparado_e_refaz_com_sobrescrever(zip_kaggle, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False)
    imagem = next((destino / "infarto").iterdir())
    imagem.write_bytes(b"alterada")

    prepare_images(zip_kaggle, destino, verbose=False, reservar_demo=False)  # já preparado: não mexe
    assert imagem.read_bytes() == b"alterada"

    prepare_images(zip_kaggle, destino, sobrescrever=True, verbose=False, reservar_demo=False)
    assert imagem.read_bytes() != b"alterada"
    outro = destino / "nao_e_nosso.txt"
    outro.write_text("x")
    prepare_images(zip_kaggle, destino, sobrescrever=True, verbose=False, reservar_demo=False)
    assert outro.exists()  # sobrescrever só apaga o que a preparação criou


def test_zip_inexistente_explica_como_baixar(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="kaggle"):
        prepare_images(tmp_path / "nao_existe.zip", tmp_path / "images", verbose=False, reservar_demo=False)


def test_cli(zip_kaggle, tmp_path, capsys) -> None:
    main(["--zip", str(zip_kaggle), "--destino", str(tmp_path / "images")])
    saida = capsys.readouterr().out
    assert "únicas" in saida and "infarto" in saida


@pytest.fixture()
def zip_completo(tmp_path):
    """Zip com 3 imagens diferentes em cada uma das 4 classes."""
    caminho = tmp_path / "completo.zip"
    semente = 100
    with zipfile.ZipFile(caminho, "w") as z:
        for pasta in (INFARTO, HISTORICO, ANORMAL, NORMAL):
            for k in range(3):
                semente += 1
                z.writestr(f"train/{pasta}/img{k}.jpg", _jpg(semente))
    return caminho


def test_reserva_uma_imagem_de_cada_classe_fora_do_treino(zip_completo, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_completo, destino, verbose=False)
    gabarito = pd.read_csv(tmp_path / "demo_gabarito.csv")

    assert sorted(gabarito["classe"]) == ["batimento_anormal", "historico_infarto", "infarto", "normal"]
    # o nome do arquivo não revela a classe
    assert sorted(gabarito["arquivo"]) == ["ecg_1.jpg", "ecg_2.jpg", "ecg_3.jpg", "ecg_4.jpg"]
    assert sorted(p.name for p in (tmp_path / "demo").iterdir()) == sorted(gabarito["arquivo"])  # só imagens na pasta
    for classe in gabarito["classe"]:
        assert len(list((destino / classe).iterdir())) == 2  # 3 - 1 reservada
    restantes = {hashlib.md5(p.read_bytes()).hexdigest() for p in destino.rglob("*.jpg")}
    assert restantes.isdisjoint(gabarito["md5"])  # nenhuma reservada continua no treino


def test_reserva_e_reprodutivel_e_feita_uma_vez_so(zip_completo, tmp_path) -> None:
    a, b = tmp_path / "a" / "images", tmp_path / "b" / "images"
    prepare_images(zip_completo, a, verbose=False)
    prepare_images(zip_completo, b, verbose=False)
    gab_a, gab_b = pd.read_csv(a.parent / "demo_gabarito.csv"), pd.read_csv(b.parent / "demo_gabarito.csv")
    pd.testing.assert_frame_equal(gab_a, gab_b)  # mesma semente, mesmas imagens e mesma ordem

    prepare_images(zip_completo, a, verbose=False)  # já preparado: não reserva de novo
    assert sum(1 for _ in a.rglob("*.jpg")) == 12 - 4


def test_sobrescrever_refaz_a_reserva_com_as_mesmas_imagens(zip_completo, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_completo, destino, verbose=False)
    antes = pd.read_csv(tmp_path / "demo_gabarito.csv")

    prepare_images(zip_completo, destino, sobrescrever=True, verbose=False)
    depois = pd.read_csv(tmp_path / "demo_gabarito.csv")

    pd.testing.assert_frame_equal(antes, depois)
    assert sum(1 for _ in destino.rglob("*.jpg")) == 12 - 4 and len(list((tmp_path / "demo").iterdir())) == 4


def test_reserva_pode_ser_feita_depois_numa_pasta_ja_preparada(zip_completo, tmp_path) -> None:
    destino = tmp_path / "images"
    prepare_images(zip_completo, destino, verbose=False, reservar_demo=False)
    assert not (tmp_path / "demo_gabarito.csv").exists()

    gabarito = reserve_demo(destino, verbose=False)
    assert len(gabarito) == 4 and (tmp_path / "demo_gabarito.csv").exists()


def test_cli_sem_demo(zip_completo, tmp_path) -> None:
    main(["--zip", str(zip_completo), "--destino", str(tmp_path / "images"), "--sem-demo"])
    assert not (tmp_path / "demo").exists()
