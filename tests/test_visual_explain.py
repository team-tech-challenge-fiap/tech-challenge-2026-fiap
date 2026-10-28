"""Testes do Grad-CAM e da análise de erros das imagens (CNN-08), com imagens sintéticas."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from cardio_visual.data.preprocessing import (  # noqa: E402
    RecorteGrade,
    class_weights,
    get_transforms,
    load_image_splits,
    make_loaders,
)
from cardio_visual.models.estimators import get_model, gradcam_layer  # noqa: E402
from cardio_visual.models.explain import (  # noqa: E402
    confusion_pairs,
    desnormalizar,
    errors_table,
    explain_image,
    grad_cam,
    sample_for_grid,
    save_gradcam_grid,
    sobrepor,
)
from cardio_visual.models.train import set_seed, train_model  # noqa: E402

CLASSES = ["sem_objeto", "esquerda", "direita"]
LADO = 64


def _imagem(classe: int, semente: int) -> Image.Image:
    """Ruído de fundo; a classe 1 tem um quadrado brilhante à esquerda e a 2, à direita (a 0 não tem)."""
    pixels = np.random.default_rng(semente).normal(60, 10, (LADO, LADO))
    if classe == 1:
        pixels[22:42, 4:24] = 230
    elif classe == 2:
        pixels[22:42, 40:60] = 230
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def modelo_treinado(tmp_path_factory):
    """CNN simples treinada de verdade no problema sintético (a posição do quadrado define a classe)."""
    raiz = tmp_path_factory.mktemp("cam") / "imagens"
    semente = 0
    for i, nome in enumerate(CLASSES):
        (raiz / nome).mkdir(parents=True)
        for k in range(30):
            semente += 1
            _imagem(i, semente).save(raiz / nome / f"{nome}_{k}.png")
    splits = load_image_splits(raiz, save=False)
    loaders = make_loaders(splits, batch_size=16, size=LADO)
    set_seed()
    resultado = train_model(get_model("cnn_simples", 3, pretrained=False), loaders, splits.classes,
                            class_weights(splits.train, splits.classes), epochs=15, lr=5e-3, verbose=False)
    return resultado.model, splits, loaders


def _tensor(img: Image.Image) -> torch.Tensor:
    return get_transforms(train=False, size=LADO)(img.convert("RGB")).unsqueeze(0)


def test_mapa_tem_o_tamanho_da_imagem_e_valores_entre_0_e_1() -> None:
    modelo = get_model("cnn_simples", 3, pretrained=False)
    mapa, classe, prob = grad_cam(modelo, _tensor(_imagem(1, 1)))

    assert mapa.shape == (LADO, LADO) and 0.0 <= mapa.min() and mapa.max() <= 1.0
    assert classe in (0, 1, 2) and 0.0 <= prob <= 1.0


def test_por_padrao_explica_a_classe_prevista_e_aceita_outra() -> None:
    modelo = get_model("cnn_simples", 3, pretrained=False).eval()
    x = _tensor(_imagem(2, 2))
    previsto = int(modelo(x).argmax())

    assert grad_cam(modelo, x)[1] == previsto
    outra = (previsto + 1) % 3
    assert grad_cam(modelo, x, classe=outra)[1] == outra


def test_nao_deixa_ganchos_pendurados_no_modelo() -> None:
    modelo = get_model("cnn_simples", 3, pretrained=False)
    camada = gradcam_layer(modelo)
    grad_cam(modelo, _tensor(_imagem(0, 3)))
    assert len(camada._forward_hooks) == 0


def test_funciona_com_o_corpo_do_resnet_congelado() -> None:
    modelo = get_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    mapa, _, _ = grad_cam(modelo, _tensor(_imagem(1, 4)))
    assert mapa.shape == (LADO, LADO)


def test_e_deterministico() -> None:
    modelo = get_model("cnn_simples", 3, pretrained=False)
    x = _tensor(_imagem(1, 5))
    assert np.allclose(grad_cam(modelo, x)[0], grad_cam(modelo, x)[0])


def test_o_calor_cai_onde_esta_o_objeto_que_define_a_classe(modelo_treinado) -> None:
    modelo, splits, _ = modelo_treinado
    for nome, lado_do_objeto in (("esquerda", "esq"), ("direita", "dir")):
        classe = CLASSES.index(nome)  # índice usado para gerar a imagem
        mapa, previsto, _ = grad_cam(modelo, _tensor(_imagem(classe, 900 + classe)))
        assert splits.classes[previsto] == nome  # o modelo aprendeu o problema (classes em ordem alfabética)
        calor_esq, calor_dir = mapa[:, : LADO // 2].mean(), mapa[:, LADO // 2:].mean()
        assert (calor_esq > calor_dir) if lado_do_objeto == "esq" else (calor_dir > calor_esq)


def test_desnormalizar_recupera_a_imagem() -> None:
    from torchvision import transforms

    img = _imagem(1, 6).convert("RGB")
    # mesmas etapas da validação (recorte da grade e redimensionamento), só que sem normalizar
    sem_normalizar = transforms.Compose([RecorteGrade(), transforms.Resize((LADO, LADO)), transforms.ToTensor()])
    esperado = sem_normalizar(img).permute(1, 2, 0).numpy()
    assert np.allclose(desnormalizar(_tensor(img)[0]), esperado, atol=1e-5)


def test_sobrepor_mantem_formato_e_intervalo() -> None:
    saida = sobrepor(np.full((8, 8, 3), 0.5), np.linspace(0, 1, 64).reshape(8, 8))
    assert saida.shape == (8, 8, 3) and saida.min() >= 0 and saida.max() <= 1


def test_tabela_de_erros_ordena_pelos_mais_confiantes() -> None:
    imagens = pd.DataFrame({"caminho": ["a", "b", "c", "d"]})
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([[0.9, 0.05, 0.05],   # acerto
                       [0.2, 0.1, 0.7],     # erro: previu 2 com 70%
                       [0.1, 0.8, 0.1],     # erro: previu 1 com 80%
                       [0.4, 0.35, 0.25]])  # acerto
    tabela = errors_table(imagens, y_true, y_prob, CLASSES)

    assert tabela["caminho"].tolist() == ["c", "b"]
    assert tabela.loc[0, "real"] == "direita" and tabela.loc[0, "previsto"] == "esquerda"
    assert tabela.loc[0, "confianca"] == pytest.approx(0.8) and tabela.loc[0, "prob_real"] == pytest.approx(0.1)


def test_tabela_de_erros_vazia_quando_nao_ha_erros() -> None:
    y = np.array([0, 1])
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert errors_table(pd.DataFrame({"caminho": ["a", "b"]}), y, p, CLASSES).empty


def test_pares_de_confusao_contam_os_erros_mais_frequentes() -> None:
    pares = confusion_pairs([0, 0, 0, 1, 2], [1, 1, 0, 1, 0], CLASSES)

    assert pares.iloc[0].tolist() == ["sem_objeto", "esquerda", 2]
    assert confusion_pairs([0, 1], [0, 1], CLASSES).empty


def test_amostra_para_a_grade_separa_acertos_e_erros() -> None:
    imagens = pd.DataFrame({"caminho": [f"i{k}" for k in range(8)]})
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    previsto = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_prob = np.eye(3)[previsto]
    amostra = sample_for_grid(imagens, y_true, y_prob, CLASSES, por_classe=1)

    assert amostra.groupby(["real", "acertou"]).size().max() == 1  # no máximo 1 por classe e tipo
    assert set(amostra["real"]) == {"sem_objeto", "esquerda"}  # a classe "direita" não tem imagens
    assert amostra["acertou"].tolist().count(False) == 2


def test_explain_image_e_a_grade_salvam_resultados(modelo_treinado, tmp_path) -> None:
    modelo, splits, loaders = modelo_treinado
    caminho = splits.val["caminho"].iloc[0]
    e = explain_image(modelo, caminho, splits.classes, size=LADO)

    assert e["sobreposta"].shape == (LADO, LADO, 3) and e["previsto"] in splits.classes

    amostra = pd.DataFrame({"caminho": splits.val["caminho"].iloc[:3], "real": "x", "previsto": "y",
                            "confianca": 0.9, "acertou": [True, False, True]})
    figura = save_gradcam_grid(modelo, amostra, splits.classes, tmp_path / "g" / "grade.png", size=LADO, colunas=2)
    assert figura.exists() and figura.stat().st_size > 0
