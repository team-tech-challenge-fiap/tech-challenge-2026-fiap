"""Gráficos reutilizáveis (EDA e avaliação), salvos em results/figures/.

TODO equipe: mover para cá os gráficos dos notebooks que forem para o relatório.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos; evita abrir janela (testes, Docker, scripts)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

CLASS_LABELS = ["Sem doença", "Doença cardiovascular"]


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> Path:
    """Salva a matriz de confusão (contagens) com os rótulos clínicos dos dois grupos.

    Args:
        y_true: alvo real (0 = sem doença, 1 = doença).
        y_pred: classes previstas.
        path: arquivo ``.png`` de saída (a pasta é criada se não existir).
        title: título do gráfico.

    Returns:
        O mesmo ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=CLASS_LABELS, cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_curve(curve, chosen: float, path: Path, title: str, min_recall: float | None = None) -> Path:
    """Salva o gráfico de recall, precision e F1 em função do limiar de decisão (T22).

    Args:
        curve: saída de ``cardio.models.evaluate.threshold_curve()``.
        chosen: limiar escolhido (linha vertical tracejada); 0,5 também é marcado, para comparação.
        path: arquivo ``.png`` de saída (a pasta é criada se não existir).
        title: título do gráfico.
        min_recall: se informado, desenha a linha horizontal do recall mínimo exigido.

    Returns:
        O mesmo ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for coluna, rotulo in (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")):
        ax.plot(curve["limiar"], curve[coluna], label=rotulo, lw=2)
    ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Limiar padrão (0,50)")
    ax.axvline(chosen, color="black", ls="--", lw=1.5, label=f"Limiar escolhido ({chosen:.2f})".replace(".", ","))
    if min_recall is not None:
        ax.axhline(min_recall, color="tab:blue", ls=":", lw=1,
                   label=f"Recall mínimo ({min_recall:.2f})".replace(".", ","))
    ax.set(title=title, xlabel="Limiar de decisão (probabilidade)", ylabel="Métrica na validação", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
