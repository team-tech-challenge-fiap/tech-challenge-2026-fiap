"""Gráficos reutilizáveis (EDA e avaliação), salvos em results/figures/.

TODO equipe: mover para cá os gráficos dos notebooks que forem para o relatório.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos; evita abrir janela (testes, Docker, scripts)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CLASS_LABELS = ["Sem doença", "Doença cardiovascular"]


def _salvar(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(probabilities: dict, y_true, path: Path, title: str = "Curvas ROC (validação)") -> Path:
    """Curvas ROC de vários modelos no mesmo gráfico, com a AUC na legenda e a diagonal do acaso.

    Args:
        probabilities: ``{nome: probabilidade da classe 1}``, calculadas na **validação**.
        y_true: alvo real.
        path: arquivo ``.png`` de saída.
        title: título do gráfico.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Baseline / acaso (AUC = 0,500)")
    ax.set(title=title, xlabel="Taxa de falsos positivos (1 − especificidade)",
           ylabel="Taxa de verdadeiros positivos (recall)")
    ax.legend(loc="lower right")
    return _salvar(fig, path)


def plot_pr_curves(probabilities: dict, y_true, path: Path,
                   title: str = "Curvas Precision-Recall (validação)") -> Path:
    """Curvas Precision-Recall de vários modelos, com a AP (área sob a curva) na legenda.

    A linha tracejada é o baseline: um modelo sem poder de previsão tem precision igual à
    prevalência da doença (~50%) em qualquer recall.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, label=f"{nome} (AP = {average_precision_score(y_true, proba):.3f})")
    ax.axhline(float(y_true.mean()), color="k", linestyle="--", label="Baseline / acaso")
    ax.set(title=title, xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left")
    return _salvar(fig, path)


def plot_calibration(proba, y_true, path: Path, name: str, n_bins: int = 10) -> Path:
    """Curva de calibração: a probabilidade prevista bate com a frequência real de doença?

    Pontos sobre a diagonal indicam um modelo calibrado; abaixo dela, o modelo exagera o risco.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fracao_real, media_prevista = calibration_curve(y_true, proba, n_bins=n_bins)
    ax.plot([0, 1], [0, 1], "k:", label="Calibração perfeita")
    ax.plot(media_prevista, fracao_real, "s-", label=name)
    ax.legend(loc="lower right")
    ax.set(title=f"Calibração — {name} (validação)", xlabel="Probabilidade prevista",
           ylabel="Fração real de pacientes com doença")
    return _salvar(fig, path)


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
