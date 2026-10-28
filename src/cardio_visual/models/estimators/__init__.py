"""Um arquivo por modelo de imagem (CNN-05), no mesmo padrão de ``cardio.models.estimators``.

Arquivos:
    cnn_simples.py   CNN pequena treinada do zero (baseline)
    resnet18.py      ResNet18 pré-treinada no ImageNet (transfer learning)

Contrato: cada arquivo expõe ``build_model(n_classes, pretrained=..., **opções)``, que devolve um ``nn.Module``
que recebe tensores ``(N, 3, lado, lado)`` e devolve **logits** ``(N, n_classes)`` (a ``CrossEntropyLoss``
aplica o softmax). Como terminam em pooling global, aceitam qualquer tamanho de imagem.

O nome do arquivo é o valor de ``--modelo`` em ``python -m cardio_visual.models.train``.
"""

from importlib import import_module

from torch import nn

MODELOS = ("cnn_simples", "resnet18")


def get_model(nome: str, n_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
    """Cria um modelo pelo nome do arquivo, como o ``--modelo`` da linha de comando.

    Args:
        nome: ``"cnn_simples"`` ou ``"resnet18"``.
        n_classes: número de classes do dataset.
        pretrained: usar pesos pré-treinados, quando o modelo tiver.
        **kwargs: opções do modelo (ex.: ``freeze_backbone=True``).

    Raises:
        ValueError: se o nome não existir.
    """
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome!r}. Opções: {', '.join(MODELOS)}")
    return import_module(f"{__name__}.{nome}").build_model(n_classes, pretrained=pretrained, **kwargs)


def gradcam_layer(modelo: nn.Module) -> nn.Module:
    """Última camada convolucional do modelo, usada pelo Grad-CAM (CNN-08): ela ainda guarda a posição espacial."""
    from .cnn_simples import CnnSimples

    if isinstance(modelo, CnnSimples):
        return modelo.blocos[-1][2]  # ReLU do último bloco, antes do pooling
    if hasattr(modelo, "layer4"):
        return modelo.layer4[-1]
    raise ValueError(f"Sem camada alvo definida para {type(modelo).__name__}")


def count_parameters(modelo: nn.Module, only_trainable: bool = False) -> int:
    """Número de parâmetros do modelo (ou só dos treináveis)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad or not only_trainable)
