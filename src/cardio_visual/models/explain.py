"""Interpretação dos modelos de imagem com Grad-CAM e análise de erros (CNN-08).

O Grad-CAM mostra **em que região da imagem** o modelo se apoiou para decidir: é o equivalente, para imagens, da
feature importance/SHAP dos modelos tabulares. Ele pega a última camada convolucional, pesa cada mapa de
características pela média do gradiente da classe escolhida e soma; o resultado é um mapa de calor da posição.

Num ECG, o esperado é que o modelo olhe para o **traçado** (QRS, segmento ST, onda T). Se o calor cair na **grade,
na legenda ou em texto**, o modelo aprendeu um atalho do dataset, e não a fisiologia.

Limite importante: o Grad-CAM é uma **pista visual**, de baixa resolução, e não prova que o modelo "entendeu" a doença.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ..config import IMAGE_SIZE, SEED
from ..data.preprocessing import DESVIO_IMAGENET, MEDIA_IMAGENET, get_transforms
from .estimators import gradcam_layer


def grad_cam(
    model: nn.Module, entrada: torch.Tensor, camada: nn.Module | None = None, classe: int | None = None
) -> tuple[np.ndarray, int, float]:
    """Mapa de calor Grad-CAM de uma imagem.

    Args:
        model: modelo treinado (é colocado em modo de avaliação).
        entrada: tensor ``(1, 3, H, W)`` já com as transformações da validação.
        camada: camada convolucional a explicar; por padrão, ``models.estimators.gradcam_layer(model)``.
        classe: índice da classe a explicar; por padrão, a classe prevista pelo modelo.

    Returns:
        ``(mapa, classe, probabilidade)``: mapa ``(H, W)`` em [0, 1], a classe explicada e a probabilidade que o
        modelo deu a ela.
    """
    model.eval()
    camada = camada if camada is not None else gradcam_layer(model)
    guardado: dict[str, torch.Tensor] = {}

    def ao_passar(_, __, saida):
        guardado["ativacao"] = saida
        saida.register_hook(lambda grad: guardado.__setitem__("gradiente", grad))

    gancho = camada.register_forward_hook(ao_passar)
    try:
        entrada = entrada.detach().clone().requires_grad_(True)  # garante o grafo mesmo com o corpo congelado
        logits = model(entrada)
        probabilidade = torch.softmax(logits, dim=1)[0].detach()
        classe = int(logits.argmax(dim=1)) if classe is None else int(classe)
        model.zero_grad()
        logits[0, classe].backward()
    finally:
        gancho.remove()  # não deixa gancho pendurado no modelo

    pesos = guardado["gradiente"].mean(dim=(2, 3), keepdim=True)
    mapa = F.relu((pesos * guardado["ativacao"]).sum(dim=1, keepdim=True))
    mapa = F.interpolate(mapa, size=entrada.shape[2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    if mapa.max() > 0:
        mapa = mapa / mapa.max()
    return mapa, classe, float(probabilidade[classe])


def desnormalizar(tensor: torch.Tensor) -> np.ndarray:
    """Desfaz a normalização do ImageNet: tensor ``(3, H, W)`` -> imagem ``(H, W, 3)`` em [0, 1]."""
    media = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    desvio = torch.tensor(DESVIO_IMAGENET).view(3, 1, 1)
    return (tensor.detach().cpu() * desvio + media).clamp(0, 1).permute(1, 2, 0).numpy()


def sobrepor(imagem: np.ndarray, mapa: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Desenha o mapa de calor (escala ``jet``) sobre a imagem ``(H, W, 3)`` em [0, 1] e devolve o mesmo formato."""
    calor = matplotlib.colormaps["jet"](mapa)[..., :3]
    return np.clip((1 - alpha) * imagem + alpha * calor, 0, 1)


def explain_image(
    model: nn.Module, caminho: Path, classes: list[str], size: int = IMAGE_SIZE, classe: int | None = None
) -> dict:
    """Grad-CAM de um arquivo de imagem, com as mesmas transformações da validação.

    Returns:
        ``{"previsto", "probabilidade", "mapa", "imagem", "sobreposta"}``; ``previsto`` é o nome da classe explicada.
    """
    with Image.open(caminho) as img:
        tensor = get_transforms(train=False, size=size)(img.convert("RGB")).unsqueeze(0)
    mapa, indice, probabilidade = grad_cam(model, tensor, classe=classe)
    imagem = desnormalizar(tensor[0])
    return {"previsto": classes[indice], "probabilidade": probabilidade, "mapa": mapa, "imagem": imagem,
            "sobreposta": sobrepor(imagem, mapa)}


def errors_table(imagens: pd.DataFrame, y_true, y_prob, classes: list[str]) -> pd.DataFrame:
    """Imagens classificadas errado, das mais confiantes às menos.

    Os erros **mais confiantes** são os mais reveladores: o modelo "tinha certeza" e errou, o que costuma indicar rótulo
    duvidoso, atalho do dataset ou uma confusão sistemática entre duas classes.

    Args:
        imagens: tabela com a coluna ``caminho``, na mesma ordem de ``y_true``.
        y_true: rótulos reais (índices). y_prob: probabilidades ``(N, n_classes)``.

    Returns:
        DataFrame com ``caminho``, ``real``, ``previsto``, ``confianca`` (probabilidade da classe prevista) e
        ``prob_real`` (probabilidade que o modelo deu à classe correta).
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    previsto = y_prob.argmax(axis=1)
    errados = np.flatnonzero(previsto != y_true)
    tabela = pd.DataFrame({
        "caminho": imagens["caminho"].to_numpy()[errados],
        "real": [classes[i] for i in y_true[errados]],
        "previsto": [classes[i] for i in previsto[errados]],
        "confianca": y_prob[errados, previsto[errados]],
        "prob_real": y_prob[errados, y_true[errados]],
    })
    return tabela.sort_values("confianca", ascending=False).reset_index(drop=True)


def confusion_pairs(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    """Pares (real -> previsto) mais frequentes entre os erros, para ver **quais classes o modelo confunde**."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    erro = y_true != y_pred
    pares = pd.DataFrame({"real": [classes[i] for i in y_true[erro]], "previsto": [classes[i] for i in y_pred[erro]]})
    if pares.empty:
        return pd.DataFrame(columns=["real", "previsto", "quantidade"])
    return pares.value_counts().rename("quantidade").reset_index()


def sample_for_grid(
    imagens: pd.DataFrame, y_true, y_prob, classes: list[str], por_classe: int = 2, seed: int = SEED
) -> pd.DataFrame:
    """Escolhe, por classe real, até ``por_classe`` acertos e ``por_classe`` erros para a grade do Grad-CAM."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    previsto = y_prob.argmax(axis=1)
    rng = np.random.default_rng(seed)
    linhas = []
    for i, nome in enumerate(classes):
        for acertou in (True, False):
            candidatos = np.flatnonzero((y_true == i) & ((previsto == y_true) == acertou))
            for j in rng.permutation(candidatos)[:por_classe]:
                linhas.append({"caminho": imagens["caminho"].iloc[j], "real": nome, "previsto": classes[previsto[j]],
                               "confianca": float(y_prob[j, previsto[j]]), "acertou": acertou})
    return pd.DataFrame(linhas, columns=["caminho", "real", "previsto", "confianca", "acertou"])


def save_gradcam_grid(
    model: nn.Module, amostra: pd.DataFrame, classes: list[str], path: Path, size: int = IMAGE_SIZE, colunas: int = 4
) -> Path:
    """Salva uma grade de Grad-CAM (imagem com mapa sobreposto). Título verde = acerto, vermelho = erro.

    Args:
        amostra: saída de ``sample_for_grid`` (colunas ``caminho``, ``real``, ``previsto``, ``confianca``, ``acertou``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    linhas = max(1, int(np.ceil(len(amostra) / colunas)))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.4 * colunas, 3.0 * linhas), squeeze=False)
    for eixo in eixos.ravel():
        eixo.axis("off")
    for eixo, (_, linha) in zip(eixos.ravel(), amostra.iterrows(), strict=False):
        explicacao = explain_image(model, linha["caminho"], classes, size)
        eixo.imshow(explicacao["sobreposta"])
        eixo.set_title(f"real: {linha['real']}\nprevisto: {linha['previsto']} ({linha['confianca']:.0%})", fontsize=8,
                       color="#2e7d32" if linha["acertou"] else "#c62828")
    fig.suptitle("Grad-CAM: onde o modelo olhou (verde = acerto, vermelho = erro)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
