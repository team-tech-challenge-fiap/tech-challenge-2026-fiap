"""Testes do treino dos modelos de imagem (CNN-07), com imagens sintéticas fáceis de separar."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from cardio_visual.data.preprocessing import class_weights, load_image_splits, make_loaders  # noqa: E402
from cardio_visual.models import train as treino  # noqa: E402
from cardio_visual.models.estimators import get_model  # noqa: E402
from cardio_visual.models.evaluate import evaluate_predictions, predict_loader  # noqa: E402

CLASSES = ["escura", "clara", "listrada"]
LADO = 32


def _imagem(classe: int, semente: int) -> Image.Image:
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0, 12, (LADO, LADO))
    base = {0: 40.0, 1: 210.0, 2: 125.0}[classe] + ruido
    if classe == 2:
        base[::4, :] += 90  # listras horizontais
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def raiz(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("ecg") / "imagens"
    semente = 0
    for i, nome in enumerate(CLASSES):
        (raiz / nome).mkdir(parents=True)
        for k in range(26):
            semente += 1
            _imagem(i, semente).save(raiz / nome / f"{nome}_{k}.png")
    return raiz


@pytest.fixture()
def isolado(tmp_path, monkeypatch):
    """Nada do teste é gravado em results/ ou models/ de verdade."""
    monkeypatch.setattr("cardio_visual.data.preprocessing.CNN_SPLIT_PATH", tmp_path / "split.csv")
    monkeypatch.setattr(treino, "CNN_RESULTS_PATH", tmp_path / "res.csv")
    monkeypatch.setattr(treino, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(treino, "FIGURES_DIR", tmp_path / "fig")
    return tmp_path


def _preparar(raiz):
    splits = load_image_splits(raiz, save=False)
    return splits, make_loaders(splits, batch_size=16, size=LADO)


def test_treino_supera_muito_o_acaso(raiz) -> None:
    splits, loaders = _preparar(raiz)
    r = treino.train_model(get_model("cnn_simples", 3, pretrained=False), loaders, splits.classes,
                           class_weights(splits.train, splits.classes), epochs=12, lr=5e-3, verbose=False)

    assert r.history["acuracia_val"].max() > 0.85  # o acaso seria ~0,33
    assert list(r.history.columns) == ["epoca", "loss_treino", "loss_val", "acuracia_val", "f1_macro_val",
                                       "recall_macro_val"]
    assert r.train_seconds > 0 and 1 <= r.best_epoch <= len(r.history)


def test_o_modelo_devolvido_e_o_da_melhor_epoca(raiz) -> None:
    splits, loaders = _preparar(raiz)
    r = treino.train_model(get_model("cnn_simples", 3, pretrained=False), loaders, splits.classes,
                           epochs=8, lr=5e-3, verbose=False)

    y, p = predict_loader(r.model, loaders["val"])
    f1_final = evaluate_predictions(y, p, splits.classes)["f1_macro"]
    assert f1_final == pytest.approx(r.history["f1_macro_val"].max(), abs=1e-9)  # restaurou o melhor estado


def test_early_stopping_para_quando_nao_ha_melhora(raiz) -> None:
    splits, loaders = _preparar(raiz)
    # modelo linear (sem BatchNorm, que atualiza estatísticas mesmo com lr = 0): com lr = 0 nada muda, o F1 da
    # validação fica igual e o treino deve parar depois de `patience` épocas sem melhora
    modelo = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * LADO * LADO, 3))
    r = treino.train_model(modelo, loaders, splits.classes, epochs=20, lr=0.0, patience=3, verbose=False)

    assert len(r.history) == 1 + 3 and r.best_epoch == 1


def test_treino_e_reprodutivel_com_a_mesma_seed(raiz) -> None:
    splits, loaders = _preparar(raiz)

    def rodar():
        treino.set_seed()  # a semente vale também para os pesos iniciais, criados antes do treino
        modelo = get_model("cnn_simples", 3, pretrained=False)
        return treino.train_model(modelo, loaders, splits.classes, epochs=3, lr=5e-3,
                                  verbose=False).history["loss_treino"].tolist()

    assert rodar() == pytest.approx(rodar())


def test_corpo_congelado_nao_muda_durante_o_treino(raiz) -> None:
    splits, loaders = _preparar(raiz)
    modelo = get_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    antes = modelo.conv1.weight.detach().clone()
    treino.train_model(modelo, loaders, splits.classes, epochs=1, lr=1e-2, verbose=False)

    assert torch.equal(antes, modelo.conv1.weight.detach())


def test_checkpoint_guarda_o_que_a_predicao_precisa(raiz, tmp_path) -> None:
    modelo = get_model("cnn_simples", 3, pretrained=False)
    caminho = treino.save_checkpoint(modelo, CLASSES, "cnn_simples", LADO, tmp_path / "m" / "cnn.pt")
    ck = torch.load(caminho, weights_only=False)

    assert ck["classes"] == CLASSES and ck["image_size"] == LADO and ck["modelo"] == "cnn_simples"
    novo = get_model("cnn_simples", 3, pretrained=False)
    novo.load_state_dict(ck["state_dict"])  # recarrega sem erro
    x = torch.randn(2, 3, LADO, LADO)
    assert torch.allclose(modelo.eval()(x), novo.eval()(x))


def test_cli_treina_avalia_e_grava_tudo(raiz, isolado) -> None:
    treino.main(["--modelo", "cnn_simples", "--root", str(raiz), "--epochs", "6", "--lr", "0.005",
                 "--batch-size", "16", "--image-size", str(LADO), "--no-pretrained", "--device", "cpu"])

    assert (isolado / "models" / f"cnn_cnn_simples_{LADO}.pt").exists()  # o nome leva o tamanho da imagem
    assert (isolado / "fig" / f"cnn_cm_cnn_simples_{LADO}_val.png").exists()
    linha = pd.read_csv(isolado / "res.csv").iloc[0]
    assert linha["modelo"] == f"cnn_simples_{LADO}" and linha["arquitetura"] == "cnn_simples"
    assert linha["n_treino"] > 0 and linha["f1_macro"] > 0.5
    assert linha["f1_macro_ic_baixo"] <= linha["f1_macro_ic_alto"]
    assert not bool(linha["pretrained"])


def test_cli_sample_reduz_o_treino(raiz, isolado) -> None:
    treino.main(["--modelo", "cnn_simples", "--root", str(raiz), "--epochs", "1", "--sample", "15",
                 "--image-size", str(LADO), "--no-pretrained"])
    assert pd.read_csv(isolado / "res.csv").iloc[0]["n_treino"] == 15


def test_cli_recusa_modelo_desconhecido(raiz) -> None:
    with pytest.raises(SystemExit):
        treino.main(["--modelo", "vgg99", "--root", str(raiz)])


def test_cli_e_reprodutivel_entre_execucoes(raiz, isolado) -> None:
    argumentos = ["--modelo", "cnn_simples", "--root", str(raiz), "--epochs", "3", "--lr", "0.005",
                  "--batch-size", "16", "--image-size", str(LADO), "--no-pretrained"]
    treino.main(argumentos)
    treino.main(argumentos)

    linhas = pd.read_csv(isolado / "res.csv")
    assert len(linhas) == 2
    assert linhas.loc[0, "f1_macro"] == linhas.loc[1, "f1_macro"]  # mesma semente, mesmo resultado
    assert linhas.loc[0, "melhor_epoca"] == linhas.loc[1, "melhor_epoca"]


def test_cli_roda_o_resnet18_com_corpo_congelado(raiz, isolado) -> None:
    treino.main(["--modelo", "resnet18", "--root", str(raiz), "--epochs", "1", "--freeze", "--no-pretrained",
                 "--batch-size", "16", "--image-size", str(LADO)])

    linha = pd.read_csv(isolado / "res.csv").iloc[0]
    assert linha["modelo"] == f"resnet18_{LADO}_freeze" and bool(linha["freeze"])
    assert (isolado / "models" / f"cnn_resnet18_{LADO}_freeze.pt").exists()


def test_empate_no_f1_e_desempatado_pela_menor_perda(raiz) -> None:
    splits, loaders = _preparar(raiz)
    treino.set_seed()
    r = treino.train_model(get_model("cnn_simples", 3, pretrained=False), loaders, splits.classes,
                           epochs=12, lr=5e-3, patience=12, verbose=False)
    h = r.history
    melhores = h[h["f1_macro_val"] >= h["f1_macro_val"].max() - 1e-4]
    assert len(melhores) > 1  # o problema sintético satura: há empate no F1
    assert r.best_epoch == int(melhores.loc[melhores["loss_val"].idxmin(), "epoca"])


def test_run_name_e_tamanhos_diferentes_nao_se_sobrescrevem(raiz, isolado) -> None:
    assert treino.run_name("resnet18", 448) == "resnet18_448"
    assert treino.run_name("resnet18", 224, freeze=True) == "resnet18_224_freeze"
    for lado in (24, LADO):
        treino.main(["--modelo", "cnn_simples", "--root", str(raiz), "--epochs", "1", "--image-size", str(lado),
                     "--no-pretrained"])
    assert {p.name for p in (isolado / "models").iterdir()} == {"cnn_cnn_simples_24.pt", f"cnn_cnn_simples_{LADO}.pt"}
    assert set(pd.read_csv(isolado / "res.csv")["modelo"]) == {"cnn_simples_24", f"cnn_simples_{LADO}"}
