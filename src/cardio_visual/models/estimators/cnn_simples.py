"""CNN simples treinada do zero (CNN-05): o baseline dos modelos de imagem.

Mostra o que uma rede pequena aprende sem conhecimento prévio. Com poucas centenas de imagens, deve ficar atrás do
transfer learning (``resnet18``), e essa comparação é um achado útil para o relatório.
"""

import torch
from torch import nn


class CnnSimples(nn.Module):
    """4 blocos (convolução, normalização, ReLU e pooling), pooling global, dropout e camada final."""

    def __init__(self, n_classes: int, canais: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3) -> None:
        super().__init__()
        blocos, entrada = [], 3
        for saida in canais:
            blocos.append(nn.Sequential(
                nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(saida),
                nn.ReLU(inplace=False),  # sem inplace, para o Grad-CAM conseguir guardar a ativação
                nn.MaxPool2d(2),
            ))
            entrada = saida
        self.blocos = nn.ModuleList(blocos)
        self.pooling_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(entrada, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bloco in self.blocos:
            x = bloco(x)
        return self.classificador(self.pooling_global(x))


def build_model(n_classes: int, pretrained: bool = False) -> nn.Module:
    """CNN simples, treinada do zero (``pretrained`` existe só para ter a mesma assinatura dos outros modelos)."""
    return CnnSimples(n_classes)
