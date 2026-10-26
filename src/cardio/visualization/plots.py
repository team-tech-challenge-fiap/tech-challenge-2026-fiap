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
