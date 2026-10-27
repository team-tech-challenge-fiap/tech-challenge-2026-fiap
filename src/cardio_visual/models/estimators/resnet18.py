"""ResNet18 com transfer learning (CNN-05): a aposta principal com dataset pequeno.

Rede de 18 camadas pré-treinada no ImageNet. Costuma ser bem melhor que uma rede do zero com poucas centenas de
imagens, porque reaproveita filtros de bordas e texturas já aprendidos; só a camada final é trocada para as classes
do ECG.
"""

from torch import nn
from torchvision import models


def build_model(n_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet18 com a camada final trocada para ``n_classes``.

    Args:
        n_classes: número de classes.
        pretrained: se True, carrega os pesos do ImageNet (baixa ~45 MB na primeira vez; precisa de internet).
        freeze_backbone: se True, congela o corpo da rede e treina só a camada final. É mais rápido e útil com
            poucas imagens; sem congelar (padrão), a rede inteira se ajusta às imagens médicas (fine-tuning).
    """
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for parametro in modelo.parameters():
            parametro.requires_grad = False
    modelo.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(modelo.fc.in_features, n_classes))  # criada depois: treinável
    return modelo
