"""Testes do pipeline de dados das imagens (CNN-04), com imagens sintéticas."""

import shutil

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from cardio_visual.data.preprocessing import (  # noqa: E402
    FOLHA_ECG,
    GRADE_ECG,
    RecorteGrade,
    class_weights,
    get_transforms,
    load_image_splits,
    make_loaders,
    subsample,
)

CLASSES = {"glioma": 40, "meningioma": 30, "sem_tumor": 20}  # desbalanceado de propósito


def _salvar(caminho, semente, lado=40, modo="L"):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(semente).integers(0, 256, (lado, lado), dtype=np.uint8)
    Image.fromarray(pixels).convert(modo).save(caminho)


@pytest.fixture()
def raiz(tmp_path):
    semente = 0
    for classe, n in CLASSES.items():
        for i in range(n):
            semente += 1
            _salvar(tmp_path / "imagens" / classe / f"{classe}_{i}.jpg", semente)
    return tmp_path / "imagens"


@pytest.fixture(autouse=True)
def split_em_tmp(tmp_path, monkeypatch):
    """Não grava o results/cnn_split.csv real durante os testes."""
    monkeypatch.setattr("cardio_visual.data.preprocessing.CNN_SPLIT_PATH", tmp_path / "cnn_split.csv")


def test_particoes_sem_sobreposicao_e_com_tudo(raiz) -> None:
    s = load_image_splits(raiz)
    todas = [set(d["caminho"]) for d in (s.train, s.val, s.test)]

    assert todas[0].isdisjoint(todas[1]) and todas[0].isdisjoint(todas[2]) and todas[1].isdisjoint(todas[2])
    assert sum(len(t) for t in todas) == sum(CLASSES.values())
    assert s.classes == sorted(CLASSES)


def test_proporcao_70_15_15_e_estratificacao(raiz) -> None:
    s = load_image_splits(raiz)
    total = sum(CLASSES.values())

    assert len(s.train) / total == pytest.approx(0.70, abs=0.03)
    assert len(s.val) / total == pytest.approx(0.15, abs=0.04)
    assert len(s.test) / total == pytest.approx(0.15, abs=0.04)
    for classe, n in CLASSES.items():  # a proporção de cada classe se mantém no treino
        assert (s.train["classe"] == classe).sum() / len(s.train) == pytest.approx(n / total, abs=0.05)


def test_mesma_seed_gera_o_mesmo_split_e_outra_seed_muda(raiz) -> None:
    a, b, c = load_image_splits(raiz), load_image_splits(raiz), load_image_splits(raiz, seed=7)

    pd.testing.assert_frame_equal(a.train, b.train)
    assert set(a.test["caminho"]) != set(c.test["caminho"])


def test_salva_o_split_com_caminho_relativo(raiz, tmp_path) -> None:
    load_image_splits(raiz)
    salvo = pd.read_csv(tmp_path / "cnn_split.csv")

    assert set(salvo["particao"]) == {"train", "val", "test"}
    assert not salvo["caminho"].str.contains(":").any()  # sem caminho absoluto do Windows
    assert len(salvo) == sum(CLASSES.values())


def test_remove_duplicatas_exatas_e_as_de_rotulo_conflitante(raiz) -> None:
    shutil.copy(raiz / "glioma" / "glioma_0.jpg", raiz / "glioma" / "copia_igual.jpg")  # mesmo rótulo: fica 1
    shutil.copy(raiz / "glioma" / "glioma_1.jpg", raiz / "sem_tumor" / "rotulo_errado.jpg")  # conflito: saem as 2

    s = load_image_splits(raiz)
    nomes = {c.split("/")[-1].split("\\")[-1] for d in (s.train, s.val, s.test) for c in d["caminho"]}

    assert len({"copia_igual.jpg", "glioma_0.jpg"} & nomes) == 1  # mesmo rótulo: fica uma só cópia
    assert "glioma_1.jpg" not in nomes and "rotulo_errado.jpg" not in nomes  # rótulo conflitante: saem as duas
    assert len(nomes) == sum(CLASSES.values()) - 1  # só o glioma_1 deixa de existir (a outra cópia é o arquivo extra)

    assert len(load_image_splits(raiz, remove_duplicates=False).train) > len(s.train)  # a opção desliga a limpeza


def test_aumento_de_dados_so_no_treino() -> None:
    treino = [type(t).__name__ for t in get_transforms(train=True).transforms]
    avaliacao = [type(t).__name__ for t in get_transforms(train=False).transforms]

    assert "RandomAffine" in treino and "ColorJitter" in treino
    assert not {"RandomHorizontalFlip", "RandomVerticalFlip"} & set(treino)  # inverter um ECG cria um traçado falso
    assert avaliacao == ["RecorteGrade", "Resize", "ToTensor", "Normalize"]
    assert treino[0] == "RecorteGrade"  # o recorte vem antes de tudo, no treino e na avaliação


def _folha_sintetica(largura=FOLHA_ECG[0], altura=FOLHA_ECG[1]) -> Image.Image:
    """Folha preta (o "texto" fora da grade) com a área da grade branca, na proporção do dataset."""
    a = np.zeros((altura, largura, 3), dtype=np.uint8)
    x0, y0, x1, y1 = (round(GRADE_ECG[i] * (largura if i % 2 == 0 else altura) / FOLHA_ECG[i % 2]) for i in range(4))
    a[y0:y1, x0:x1] = 255
    return Image.fromarray(a)


def test_recorte_pega_exatamente_a_grade_na_folha_original() -> None:
    recortada = RecorteGrade()(_folha_sintetica())
    x0, y0, x1, y1 = GRADE_ECG

    assert recortada.size == (x1 - x0, y1 - y0)
    assert np.asarray(recortada).min() == 255  # nada do "texto" (preto) sobrou


def test_recorte_e_proporcional_em_outra_resolucao() -> None:
    meia = _folha_sintetica(FOLHA_ECG[0] // 2, FOLHA_ECG[1] // 2)
    recortada = RecorteGrade()(meia)

    assert abs(recortada.width - (GRADE_ECG[2] - GRADE_ECG[0]) / 2) <= 1
    assert abs(recortada.height - (GRADE_ECG[3] - GRADE_ECG[1]) / 2) <= 1
    assert np.asarray(recortada).min() == 255


def test_pipeline_recorta_antes_de_redimensionar_e_pode_ser_desligado() -> None:
    folha = _folha_sintetica()
    com = get_transforms(train=False, size=64)(folha)
    sem = get_transforms(train=False, size=64, recorte=None)(folha)

    assert com.shape == sem.shape == (3, 64, 64)
    assert all(float(com[c].min()) == float(com[c].max()) for c in range(3))  # só a grade (branca): cada canal uniforme
    assert sem.min() < com.min()  # sem o recorte, o "texto" (preto) continua na imagem
    assert [type(t).__name__ for t in get_transforms(train=False, recorte=None).transforms][0] == "Resize"


@pytest.mark.parametrize("fracoes", [(0.5, 0.1, 0.4, 0.9), (0.0, 0.0, 1.2, 1.0), (-0.1, 0.0, 1.0, 1.0)])
def test_recorte_invalido_levanta_erro(fracoes) -> None:
    with pytest.raises(ValueError):
        RecorteGrade(fracoes)


def test_make_loaders_aplica_o_mesmo_recorte_nas_tres_particoes(raiz) -> None:
    lotes = make_loaders(load_image_splits(raiz), batch_size=4, size=16, recorte=(0.25, 0.25, 0.75, 0.75))
    for particao in ("train", "val", "test"):
        assert lotes[particao].dataset.transform.transforms[0].fracoes == (0.25, 0.25, 0.75, 0.75)


def test_loaders_devolvem_lotes_com_3_canais_mesmo_em_cinza(raiz) -> None:
    s = load_image_splits(raiz)
    lotes = make_loaders(s, batch_size=8, size=32)
    x, y = next(iter(lotes["train"]))

    assert x.shape == (8, 3, 32, 32) and y.shape == (8,)
    assert set(y.tolist()) <= {0, 1, 2}
    assert len(lotes["val"].dataset) == len(s.val) and len(lotes["test"].dataset) == len(s.test)


def test_imagem_rgb_tambem_funciona(tmp_path) -> None:
    for deslocamento, classe in enumerate(("a", "b")):
        for i in range(10):
            _salvar(tmp_path / "rgb" / classe / f"{i}.png", 100 * deslocamento + i, modo="RGB")
    x, _ = next(iter(make_loaders(load_image_splits(tmp_path / "rgb"), batch_size=4, size=16)["val"]))
    assert x.shape[1:] == (3, 16, 16)


def test_ordem_do_treino_e_reprodutivel(raiz) -> None:
    s = load_image_splits(raiz)
    a = [y.tolist() for _, y in make_loaders(s, batch_size=16, size=16)["train"]]
    b = [y.tolist() for _, y in make_loaders(s, batch_size=16, size=16)["train"]]
    assert a == b


def test_class_weights_dao_mais_peso_a_classe_rara(raiz) -> None:
    s = load_image_splits(raiz)
    pesos = class_weights(s.train, s.classes)

    assert pesos.shape == (3,)
    assert pesos[s.classes.index("sem_tumor")] > pesos[s.classes.index("glioma")]
    equilibrado = pd.DataFrame({"classe": ["a"] * 10 + ["b"] * 10})
    assert class_weights(equilibrado, ["a", "b"]).tolist() == [1.0, 1.0]


def test_subsample_mantem_a_estratificacao(raiz) -> None:
    s = load_image_splits(raiz)
    amostra = subsample(s.train, 20)

    assert len(amostra) == 20 and set(amostra["classe"]) == set(CLASSES)
    assert subsample(s.train, 10_000) is s.train
