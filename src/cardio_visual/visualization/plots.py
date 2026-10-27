"""Gráficos dos modelos de imagem (matriz de confusão e curvas ROC), salvos em results/figures/.

Mesmo papel do ``cardio.visualization.plots`` no projeto tabular.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos; evita abrir janela (testes, Docker, scripts)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_confusion_matrix(y_true, y_pred, classes: list[str], path: Path, title: str) -> Path:
    """Salva a matriz de confusão (contagens) com os nomes das classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 1.4 * len(classes) + 2.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(len(classes))), display_labels=classes,
        cmap="Blues", values_format="d", ax=ax, xticks_rotation=20,
    )
    ax.set(title=title, xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_per_class(y_true, y_prob, classes: list[str], path: Path, title: str) -> Path:
    """Curva ROC de cada classe contra as demais, com a AUC na legenda e a diagonal do acaso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, nome in enumerate(classes):
        alvo = (y_true == i).astype(int)
        if alvo.min() == alvo.max():  # a classe não aparece (ou só ela aparece): não há curva
            continue
        fpr, tpr, _ = roc_curve(alvo, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc_score(alvo, y_prob[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Acaso (AUC = 0,500)")
    ax.set(title=title, xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos (recall)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
