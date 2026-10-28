"""Testes da predição, do teste cego e da avaliação final no teste (CNN-09), com imagens sintéticas."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from cardio_visual.data.preprocessing import class_weights, load_image_splits, make_loaders  # noqa: E402
from cardio_visual.models import final_model as final  # noqa: E402
from cardio_visual.models import predict  # noqa: E402
from cardio_visual.models.estimators import get_model  # noqa: E402
from cardio_visual.models.train import save_checkpoint, set_seed, train_model  # noqa: E402

CLASSES = ["arritmia", "infarto", "normal"]  # ordem alfabética, como o load_image_splits devolve
LADO = 32


def _imagem(classe: str, semente: int) -> Image.Image:
    rng = np.random.default_rng(semente)
    base = {"arritmia": 40.0, "infarto": 125.0, "normal": 210.0}[classe] + rng.normal(0, 12, (LADO, LADO))
    if classe == "infarto":
        base[::4, :] += 90
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")


@pytest.fixture(scope="module")
def ambiente(tmp_path_factory):
    """Dataset sintético, um modelo treinado e salvo, e uma pasta de teste cego com gabarito."""
    base = tmp_path_factory.mktemp("pred")
    raiz = base / "images"
    semente = 0
    for classe in CLASSES:
        (raiz / classe).mkdir(parents=True)
        for k in range(24):
            semente += 1
            _imagem(classe, semente).save(raiz / classe / f"{classe}_{k}.png")
    demo = base / "demo"
    demo.mkdir()
    linhas = []
    for i, classe in enumerate(["normal", "arritmia", "infarto"], start=1):
        _imagem(classe, 1000 + i).save(demo / f"ecg_{i}.jpg")
        linhas.append({"arquivo": f"ecg_{i}.jpg", "classe": classe, "md5": "x", "origem": "x"})
    pd.DataFrame(linhas).to_csv(base / "demo_gabarito.csv", index=False)

    splits = load_image_splits(raiz, save=False)
    set_seed()
    modelo = train_model(get_model("cnn_simples", 3, pretrained=False), make_loaders(splits, 16, LADO), splits.classes,
                         class_weights(splits.train, splits.classes), epochs=8, lr=5e-3, verbose=False).model
    checkpoint = save_checkpoint(modelo, splits.classes, "cnn_simples", LADO,
                                 base / "models" / f"cnn_cnn_simples_{LADO}.pt")
    return {"base": base, "raiz": raiz, "demo": demo, "checkpoint": checkpoint, "modelo": modelo}


def test_load_checkpoint_recria_o_mesmo_modelo(ambiente) -> None:
    ck = predict.load_checkpoint(ambiente["checkpoint"])
    x = torch.randn(2, 3, LADO, LADO)

    assert ck["classes"] == CLASSES and ck["image_size"] == LADO and ck["modelo"] == "cnn_simples"
    assert not ck["model"].training
    assert torch.allclose(ck["model"](x), ambiente["modelo"].eval()(x))


def test_checkpoint_inexistente_explica_como_gerar(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        predict.load_checkpoint(tmp_path / "nao_existe.pt")


def test_predict_image_devolve_probabilidades_e_o_aviso(ambiente) -> None:
    ck = predict.load_checkpoint(ambiente["checkpoint"])
    r = predict.predict_image(ck, ambiente["demo"] / "ecg_1.jpg")

    assert set(r["probabilidades"]) == set(CLASSES)
    assert sum(r["probabilidades"].values()) == pytest.approx(1.0, abs=1e-5)
    assert r["classe"] == max(r["probabilidades"], key=r["probabilidades"].get)
    assert r["prob_alterado"] == pytest.approx(1 - r["probabilidades"]["normal"])
    assert "médico" in r["aviso"]


def test_aceita_imagem_pil_e_caminho_com_o_mesmo_resultado(ambiente) -> None:
    ck = predict.load_checkpoint(ambiente["checkpoint"])
    caminho = ambiente["demo"] / "ecg_2.jpg"
    with Image.open(caminho) as img:
        pil = predict.predict_image(ck, img.copy())
    assert pil["probabilidades"] == pytest.approx(predict.predict_image(ck, caminho)["probabilidades"])


def test_teste_cego_confere_com_o_gabarito(ambiente) -> None:
    ck = predict.load_checkpoint(ambiente["checkpoint"])
    arquivos = sorted(ambiente["demo"].glob("*.jpg"))
    tabela = predict.check_demo(predict.predict_files(ck, arquivos), ambiente["base"] / "demo_gabarito.csv")

    assert tabela["arquivo"].tolist() == ["ecg_1.jpg", "ecg_2.jpg", "ecg_3.jpg"]
    assert tabela["classe_real"].tolist() == ["normal", "arritmia", "infarto"]
    assert tabela["acertou"].sum() >= 2  # problema sintético fácil


def test_cli_demo_sem_gabarito_nao_mostra_a_resposta(ambiente, capsys) -> None:
    predict.main(["--demo", "--demo-dir", str(ambiente["demo"]), "--checkpoint", str(ambiente["checkpoint"])])
    saida = capsys.readouterr().out
    assert "ecg_1.jpg" in saida and "classe_real" not in saida and "Acertos" not in saida


def test_cli_demo_com_gabarito(ambiente, capsys) -> None:
    predict.main(["--demo", "--gabarito", "--demo-dir", str(ambiente["demo"]),
                  "--checkpoint", str(ambiente["checkpoint"])])
    saida = capsys.readouterr().out
    assert "classe_real" in saida and "Acertos:" in saida and "médico" in saida


def test_cli_sem_imagens_da_erro(ambiente) -> None:
    with pytest.raises(SystemExit):
        predict.main(["--checkpoint", str(ambiente["checkpoint"])])


def test_metricas_binarias_calculadas_a_mao() -> None:
    # classes: arritmia (0), infarto (1), normal (2); alterado = 0 ou 1
    y = np.array([0, 1, 2, 2, 1])
    p = np.array([[0.7, 0.2, 0.1],    # alterado, previsto alterado (P normal 0,1)
                  [0.1, 0.2, 0.7],    # alterado, previsto normal -> falso negativo
                  [0.1, 0.1, 0.8],    # normal, previsto normal
                  [0.5, 0.3, 0.2],    # normal, previsto alterado -> falso positivo
                  [0.2, 0.6, 0.2]])   # alterado, previsto alterado
    m = final.binary_metrics(y, p, CLASSES)

    assert m["recall_alterado"] == pytest.approx(2 / 3)
    assert m["especificidade"] == pytest.approx(1 / 2)
    assert m["precision_alterado"] == pytest.approx(2 / 3)
    assert m["acuracia_binaria"] == pytest.approx(3 / 5)


def test_avaliacao_final_grava_resultados_e_figuras_do_teste(ambiente, tmp_path) -> None:
    r = final.evaluate_final(ambiente["checkpoint"], ambiente["raiz"], tmp_path / "fig", tmp_path / "res.csv",
                             tmp_path / "por_classe.csv")
    linha = pd.read_csv(tmp_path / "res.csv")

    assert len(linha) == 1 and linha.loc[0, "modelo"] == f"cnn_simples_{LADO}"
    assert linha.loc[0, "n_teste"] == len(load_image_splits(ambiente["raiz"], save=False).test)
    assert {"recall_alterado", "especificidade", "auc_binaria", "f1_macro_ic_baixo"} <= set(linha.columns)
    assert (tmp_path / "fig" / f"cnn_cm_cnn_simples_{LADO}_test.png").exists()
    assert len(pd.read_csv(tmp_path / "por_classe.csv")) == 3 and r["f1_macro"] > 0.5


def test_avaliacao_final_recusa_checkpoint_de_outras_classes(ambiente, tmp_path) -> None:
    outro = save_checkpoint(get_model("cnn_simples", 2, pretrained=False), ["a", "b"], "cnn_simples", LADO,
                            tmp_path / "outro.pt")
    with pytest.raises(ValueError, match="classes"):
        final.evaluate_final(outro, ambiente["raiz"], tmp_path, tmp_path / "r.csv", tmp_path / "c.csv")
