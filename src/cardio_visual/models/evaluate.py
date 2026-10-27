"""Avaliação padronizada dos modelos de imagem (CNN-06).

Por que estas métricas (triagem de ECG: infarto e arritmia não podem passar despercebidos):

- **Recall por classe**: de todos os ECGs de uma condição, quantos o modelo reconhece. O falso negativo
  (ex.: infarto classificado como normal) é o erro mais grave, então o recall é a métrica principal, olhada
  **por classe** e na média entre as classes (macro).
- **Precision**: dos ECGs que o modelo atribuiu a uma classe, quantos realmente são dela.
- **F1**: média harmônica de precision e recall.
- **AUC (um-contra-todos)**: capacidade de separar cada classe das demais, sem depender de um limiar.
- **Accuracy**: só é informativa se as classes forem balanceadas; com classes desiguais ela esconde o erro nas pequenas.

Com datasets pequenos (centenas de imagens) a validação tem poucas dezenas de ECGs por classe, e a métrica
varia bastante de uma partição para outra. Por isso ``bootstrap_ci`` dá um **intervalo de confiança**: diferenças
menores que o intervalo entre dois modelos não devem ser tratadas como melhora real.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ..config import CNN_RESULTS_PATH, FIGURES_DIR, SEED
from ..visualization.plots import plot_confusion_matrix, plot_roc_per_class


def predict_loader(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades do modelo para todo o ``loader`` (softmax dos logits), sem calcular gradientes.

    Returns:
        ``(y_true, y_prob)``: rótulos reais ``(N,)`` e probabilidades ``(N, n_classes)``.
    """
    model = model.to(device).eval()
    verdadeiros, probabilidades = [], []
    with torch.no_grad():
        for imagens, rotulos in loader:
            logits = model(imagens.to(device))
            probabilidades.append(torch.softmax(logits, dim=1).cpu().numpy())
            verdadeiros.append(np.asarray(rotulos))
    return np.concatenate(verdadeiros), np.concatenate(probabilidades)


def _matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz ``(real, previsto)`` por contagem, rápida o bastante para o bootstrap."""
    return np.bincount(n_classes * y_true + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def _metricas_por_classe(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall e F1 de cada classe a partir da matriz de confusão (0 quando a conta não existe)."""
    vp = np.diag(cm).astype(float)
    previstos, reais = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(vp, previstos, out=np.zeros_like(vp), where=previstos > 0)
    recall = np.divide(vp, reais, out=np.zeros_like(vp), where=reais > 0)
    soma = precision + recall
    f1 = np.divide(2 * precision * recall, soma, out=np.zeros_like(vp), where=soma > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "suporte": reais}


def evaluate_predictions(y_true, y_prob, classes: list[str]) -> dict:
    """Métricas de classificação a partir das probabilidades.

    Args:
        y_true: rótulos reais (índices em ``classes``).
        y_prob: probabilidades ``(N, n_classes)``.
        classes: nomes das classes, na ordem dos índices.

    Returns:
        Dicionário com ``acuracia``, ``precision_macro``, ``recall_macro``, ``f1_macro`` e ``auc_ovr``
        (``nan`` se alguma classe não aparece em ``y_true``), mais ``por_classe``: DataFrame com ``classe``,
        ``suporte``, ``precision``,
        ``recall`` e ``f1``. As médias macro contam só as classes que aparecem em ``y_true``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(classes)
    if y_prob.ndim != 2 or y_prob.shape[1] != n:
        raise ValueError(f"y_prob deve ter {n} colunas (uma por classe); recebido: {y_prob.shape}")
    y_pred = y_prob.argmax(axis=1)
    cm = _matriz_confusao(y_true, y_pred, n)
    por_classe = _metricas_por_classe(cm)
    presentes = por_classe["suporte"] > 0

    if presentes.all():
        auc = roc_auc_score(y_true, y_prob[:, 1]) if n == 2 else roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n)))
    else:
        auc = float("nan")

    return {
        "acuracia": float((y_true == y_pred).mean()),
        "precision_macro": float(por_classe["precision"][presentes].mean()),
        "recall_macro": float(por_classe["recall"][presentes].mean()),
        "f1_macro": float(por_classe["f1"][presentes].mean()),
        "auc_ovr": float(auc),
        "por_classe": pd.DataFrame({"classe": classes, **por_classe}),
    }


def bootstrap_ci(
    y_true, y_prob, metrica: str = "recall_macro", n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float]:
    """Intervalo de confiança (percentil) de uma métrica macro, por bootstrap das imagens.

    Sorteia ``n`` imagens com reposição ``n_boot`` vezes e recalcula a métrica. Dá uma ideia de quanto o
    resultado variaria com outra amostra de validação: em datasets pequenos, o intervalo é largo.

    Args:
        metrica: ``"recall_macro"``, ``"f1_macro"``, ``"precision_macro"`` ou ``"acuracia"``.
        alpha: nível (0,05 dá o intervalo de 95%).
    """
    if metrica not in ("recall_macro", "f1_macro", "precision_macro", "acuracia"):
        raise ValueError(f"Métrica não suportada no bootstrap: {metrica!r}")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    n_classes = np.asarray(y_prob).shape[1]
    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[i], y_pred[i]
        if metrica == "acuracia":
            valores.append((yt == yp).mean())
            continue
        m = _metricas_por_classe(_matriz_confusao(yt, yp, n_classes))
        presentes = m["suporte"] > 0
        chave = metrica.removesuffix("_macro")
        valores.append(m[chave][presentes].mean())
    baixo, alto = np.percentile(valores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(baixo), float(alto)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    name: str | None = None,
    split: str = "val",
    figures_dir: Path = FIGURES_DIR,
    device: str = "cpu",
) -> dict:
    """Avalia um modelo num ``loader``: métricas, intervalos de confiança e, com ``name``, as figuras.

    Use a **validação** para escolher modelos; o **teste** só uma vez, no modelo final.

    Returns:
        Saída de ``evaluate_predictions`` mais ``recall_macro_ic`` e ``f1_macro_ic`` (tuplas) e, se ``name`` for dado,
        as figuras ``cnn_cm_<name>_<split>.png`` e ``cnn_roc_<name>_<split>.png`` em ``figures_dir``.
    """
    y_true, y_prob = predict_loader(model, loader, device)
    resultado = evaluate_predictions(y_true, y_prob, classes)
    resultado["recall_macro_ic"] = bootstrap_ci(y_true, y_prob, "recall_macro")
    resultado["f1_macro_ic"] = bootstrap_ci(y_true, y_prob, "f1_macro")
    if name is not None:
        figures_dir = Path(figures_dir)
        plot_confusion_matrix(y_true, y_prob.argmax(axis=1), classes, figures_dir / f"cnn_cm_{name}_{split}.png",
                              f"Matriz de confusão — {name} ({split})")
        plot_roc_per_class(y_true, y_prob, classes, figures_dir / f"cnn_roc_{name}_{split}.png",
                           f"Curvas ROC por classe — {name} ({split})")
    return resultado


def append_result(linha: dict, path: Path = CNN_RESULTS_PATH) -> None:
    """Acrescenta uma linha ao CSV de resultados (cria o arquivo com cabeçalho na primeira vez)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([linha]).to_csv(path, mode="a", header=not path.exists(), index=False)


def summary_table(path: Path = CNN_RESULTS_PATH, ordenar_por: str = "f1_macro") -> pd.DataFrame:
    """Resultados de todos os modelos (a linha mais recente de cada modelo), do melhor ao pior."""
    tabela = pd.read_csv(path).drop_duplicates("modelo", keep="last")
    return tabela.sort_values(ordenar_por, ascending=False).reset_index(drop=True)
