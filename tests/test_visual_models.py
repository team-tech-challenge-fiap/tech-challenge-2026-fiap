"""Testes dos modelos de imagem (CNN-05). Nenhum teste baixa pesos da internet (pretrained=False)."""

import pytest

torch = pytest.importorskip("torch")

from cardio_visual.models.estimators import (  # noqa: E402
    MODELOS,
    count_parameters,
    get_model,
    gradcam_layer,
)


@pytest.mark.parametrize("nome", list(MODELOS))
@pytest.mark.parametrize("n_classes", [2, 4])
def test_saida_tem_um_logit_por_classe(nome, n_classes) -> None:
    modelo = get_model(nome, n_classes, pretrained=False).eval()
    with torch.no_grad():
        saida = modelo(torch.randn(3, 3, 64, 64))
    assert saida.shape == (3, n_classes)


@pytest.mark.parametrize("nome", list(MODELOS))
def test_aceita_tamanhos_diferentes_de_imagem(nome) -> None:
    modelo = get_model(nome, 3, pretrained=False).eval()
    with torch.no_grad():
        assert modelo(torch.randn(2, 3, 48, 48)).shape == modelo(torch.randn(2, 3, 96, 96)).shape


def test_cnn_simples_e_bem_menor_que_a_resnet() -> None:
    assert count_parameters(get_model("cnn_simples", 4, pretrained=False)) < 200_000
    assert count_parameters(get_model("resnet18", 4, pretrained=False)) > 10_000_000


def test_freeze_backbone_treina_so_a_camada_final() -> None:
    modelo = get_model("resnet18", 4, pretrained=False, freeze_backbone=True)
    treinaveis = {n for n, p in modelo.named_parameters() if p.requires_grad}

    assert treinaveis and all(n.startswith("fc.") for n in treinaveis)
    assert count_parameters(modelo, only_trainable=True) == 512 * 4 + 4
    livre = get_model("resnet18", 4, pretrained=False)
    assert count_parameters(livre, only_trainable=True) == count_parameters(livre)


@pytest.mark.parametrize("nome", list(MODELOS))
def test_gradcam_layer_e_uma_camada_do_proprio_modelo(nome) -> None:
    modelo = get_model(nome, 3, pretrained=False)
    camada = gradcam_layer(modelo)
    assert any(camada is m for m in modelo.modules())


def test_nome_desconhecido_levanta_erro() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        get_model("vgg99", 3)


def test_state_dict_nao_tem_chaves_duplicadas() -> None:
    chaves = list(get_model("resnet18", 3, pretrained=False).state_dict())
    assert len(chaves) == len(set(chaves)) and not any("gradcam" in c for c in chaves)
