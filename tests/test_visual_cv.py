"""Testes da validação cruzada dos modelos de imagem (CNN-07), com imagens sintéticas."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from cardio_visual.data.preprocessing import load_image_splits  # noqa: E402
from cardio_visual.models import cv  # noqa: E402

LADO = 32


def _imagem(classe: int, semente: int) -> Image.Image:
    rng = np.random.default_rng(semente)
    base = {0: 40.0, 1: 210.0, 2: 125.0}[classe] + rng.normal(0, 12, (LADO, LADO))
    if classe == 2:
        base[::4, :] += 90
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def raiz(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("cv") / "imagens"
    semente = 0
    for i, nome in enumerate(["escura", "clara", "listrada"]):
        (raiz / nome).mkdir(parents=True)
        for k in range(20):
            semente += 1
            _imagem(i, semente).save(raiz / nome / f"{nome}_{k}.png")
    return raiz


@pytest.fixture()
def isolado(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "CNN_CV_RESULTS_PATH", tmp_path / "cv.csv")
    return tmp_path


def _rodar(raiz, **kwargs):
    opcoes = dict(folds=3, epochs=4, lr=5e-3, image_size=LADO, pretrained=False, verbose=False)
    opcoes.update(kwargs)
    return cv.cross_validate("cnn_simples", raiz, **opcoes)


def test_um_resultado_por_fold_com_recall_por_classe(raiz) -> None:
    r = _rodar(raiz)

    assert r["fold"].tolist() == [1, 2, 3]
    assert {"recall_escura", "recall_clara", "recall_listrada", "f1_macro", "melhor_epoca"} <= set(r.columns)
    assert r["f1_macro"].mean() > 0.6  # o problema sintético é fácil


def test_o_teste_nunca_entra_na_validacao_cruzada(raiz, monkeypatch) -> None:
    usadas = []
    original = cv.make_loaders

    def espiao(splits, *args, **kwargs):
        usadas.extend(splits.train["caminho"])
        usadas.extend(splits.val["caminho"])
        return original(splits, *args, **kwargs)

    monkeypatch.setattr(cv, "make_loaders", espiao)
    _rodar(raiz, epochs=1)
    teste = set(load_image_splits(raiz, save=False).test["caminho"])
    assert usadas and teste.isdisjoint(usadas)


def test_folds_cobrem_treino_e_validacao_sem_sobrepor(raiz, monkeypatch) -> None:
    validacoes = []
    original = cv.make_loaders

    def espiao(splits, *args, **kwargs):
        validacoes.append(set(splits.val["caminho"]))
        return original(splits, *args, **kwargs)

    monkeypatch.setattr(cv, "make_loaders", espiao)
    _rodar(raiz, epochs=1)
    splits = load_image_splits(raiz, save=False)
    assert set().union(*validacoes) == set(splits.train["caminho"]) | set(splits.val["caminho"])
    assert sum(len(v) for v in validacoes) == len(set().union(*validacoes))  # cada imagem valida em um fold só


def test_e_reprodutivel(raiz) -> None:
    pd.testing.assert_frame_equal(_rodar(raiz, epochs=2).drop(columns="tempo_s"),
                                  _rodar(raiz, epochs=2).drop(columns="tempo_s"))


def test_resumo_tem_media_e_desvio() -> None:
    r = pd.DataFrame({"fold": [1, 2], "f1_macro": [0.5, 0.7], "melhor_epoca": [3, 4], "tempo_s": [1.0, 2.0]})
    resumo = cv.summarize(r)
    assert list(resumo.index) == ["f1_macro"]
    assert resumo.loc["f1_macro", "media"] == pytest.approx(0.6)
    assert resumo.loc["f1_macro", "desvio"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_cli_grava_uma_linha_por_fold(raiz, isolado, capsys) -> None:
    cv.main(["--modelo", "cnn_simples", "--root", str(raiz), "--folds", "3", "--epochs", "1", "--image-size", str(LADO),
             "--no-pretrained"])
    linhas = pd.read_csv(isolado / "cv.csv")

    assert len(linhas) == 3 and set(linhas["modelo"]) == {f"cnn_simples_{LADO}"}
    assert "Validação cruzada" in capsys.readouterr().out
