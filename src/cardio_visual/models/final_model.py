"""Avaliação final do modelo de imagem no conjunto de teste (CNN-07/CNN-09).

Uso: ``python -m cardio_visual.models.final_model [--checkpoint models/cnn_resnet18_448.pt]``.

O teste é usado **uma única vez**, só no modelo escolhido pela validação, como na T23 do projeto tabular. Grava
``results/cnn_test_results.csv`` (uma linha), ``results/cnn_test_por_classe.csv`` e as figuras
``cnn_cm_<execução>_test.png`` e ``cnn_roc_<execução>_test.png``.

Além das 4 classes, reporta a **visão binária** (normal × alterado), decisão D2 do extra: a classe infarto tem poucas
imagens no teste, então a métrica dela é instável; juntar as alterações dá uma medida mais estável de "esse ECG precisa
de atenção?", que é a pergunta de uma triagem.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import FIGURES_DIR, IMAGES_DIR, RESULTS_DIR
from ..data.preprocessing import load_image_splits, make_loaders
from .evaluate import evaluate_model, predict_loader
from .predict import FINAL_CHECKPOINT, load_checkpoint

CNN_TEST_RESULTS_PATH = RESULTS_DIR / "cnn_test_results.csv"
CNN_TEST_POR_CLASSE_PATH = RESULTS_DIR / "cnn_test_por_classe.csv"


def binary_metrics(y_true, y_prob, classes: list[str], normal: str = "normal", limiar: float = 0.5) -> dict[str, float]:
    """Métricas da visão binária: "alterado" (qualquer classe que não seja ``normal``) × normal.

    A probabilidade de alterado é ``1 - P(normal)``; com ``limiar`` 0,5.

    Returns:
        ``recall_alterado`` (sensibilidade: alterados que o modelo sinaliza), ``especificidade`` (normais que ele não
        sinaliza), ``precision_alterado``, ``f1_alterado``, ``acuracia_binaria`` e ``auc_binaria``.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    i_normal = classes.index(normal)
    real = (y_true != i_normal).astype(int)
    prob = 1.0 - y_prob[:, i_normal]
    previsto = (prob >= limiar).astype(int)
    vp, fn = int(((real == 1) & (previsto == 1)).sum()), int(((real == 1) & (previsto == 0)).sum())
    vn, fp = int(((real == 0) & (previsto == 0)).sum()), int(((real == 0) & (previsto == 1)).sum())
    recall = vp / (vp + fn) if vp + fn else 0.0
    precision = vp / (vp + fp) if vp + fp else 0.0
    return {
        "recall_alterado": recall,
        "especificidade": vn / (vn + fp) if vn + fp else 0.0,
        "precision_alterado": precision,
        "f1_alterado": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "acuracia_binaria": (vp + vn) / len(real),
        "auc_binaria": roc_auc_score(real, prob) if 0 < real.sum() < len(real) else float("nan"),
    }


def evaluate_final(
    checkpoint_path: Path = FINAL_CHECKPOINT,
    root: Path = IMAGES_DIR,
    figures_dir: Path = FIGURES_DIR,
    results_path: Path = CNN_TEST_RESULTS_PATH,
    por_classe_path: Path = CNN_TEST_POR_CLASSE_PATH,
    device: str = "cpu",
) -> dict:
    """Avalia o checkpoint no teste (mesmo split 70/15/15 do treino) e grava os resultados.

    Returns:
        Saída de ``evaluate_model`` (4 classes, com intervalos de confiança) mais as métricas binárias.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    splits = load_image_splits(root, save=False)
    if splits.classes != checkpoint["classes"]:
        raise ValueError(f"As classes do checkpoint {checkpoint['classes']} não batem com as do dataset "
                         f"{splits.classes}")
    loader = make_loaders(splits, size=checkpoint["image_size"])["test"]
    execucao = Path(checkpoint_path).stem.removeprefix("cnn_")

    resultado = evaluate_model(checkpoint["model"], loader, splits.classes, name=execucao, split="test",
                               figures_dir=figures_dir, device=device)
    y_true, y_prob = predict_loader(checkpoint["model"], loader, device)
    binario = binary_metrics(y_true, y_prob, splits.classes)
    resultado.update(binario)

    linha = {"modelo": execucao, "n_teste": len(y_true), "acuracia": resultado["acuracia"],
             "f1_macro": resultado["f1_macro"], "recall_macro": resultado["recall_macro"],
             "precision_macro": resultado["precision_macro"], "auc_ovr": resultado["auc_ovr"],
             "recall_macro_ic_baixo": resultado["recall_macro_ic"][0],
             "recall_macro_ic_alto": resultado["recall_macro_ic"][1],
             "f1_macro_ic_baixo": resultado["f1_macro_ic"][0], "f1_macro_ic_alto": resultado["f1_macro_ic"][1],
             **binario}
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([linha]).to_csv(results_path, index=False)  # um modelo final, uma avaliação: sobrescreve
    resultado["por_classe"].to_csv(por_classe_path, index=False)
    return resultado


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Avaliação final do modelo de imagem no teste (uso único)")
    parser.add_argument("--checkpoint", type=Path, default=FINAL_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=IMAGES_DIR)
    args = parser.parse_args(argv)

    r = evaluate_final(args.checkpoint, args.root)
    print(r["por_classe"].round(3).to_string(index=False))
    print(f"\n4 classes: acurácia {r['acuracia']:.3f} | F1 macro {r['f1_macro']:.3f} "
          f"(IC 95%: {r['f1_macro_ic'][0]:.3f} a {r['f1_macro_ic'][1]:.3f}) | recall macro {r['recall_macro']:.3f}")
    print(f"Binário (normal × alterado): recall {r['recall_alterado']:.3f} | "
          f"especificidade {r['especificidade']:.3f} | "
          f"F1 {r['f1_alterado']:.3f} | AUC {r['auc_binaria']:.3f}")


if __name__ == "__main__":
    main()
