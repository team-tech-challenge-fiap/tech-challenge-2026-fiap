"""Validação cruzada estratificada dos modelos de imagem (CNN-07).

Uso: ``python -m cardio_visual.models.cv --modelo resnet18 --folds 5 --epochs 15 [--image-size 224]``.

Com poucas centenas de imagens, uma única validação (~70 imagens, ~4 de infarto) dá métricas muito ruidosas: trocar
uma imagem de partição muda o resultado. A validação cruzada junta treino e validação, divide em ``k`` partes
estratificadas e treina ``k`` vezes, cada vez validando numa parte diferente. A **média** compara modelos com menos
ruído, e o **desvio-padrão** mostra quanto o resultado depende da partição. O **teste continua de fora**: só é usado
uma vez, no modelo final (``cardio_visual.models.final_model``).
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import CNN_BATCH_SIZE, IMAGE_SIZE, IMAGES_DIR, RESULTS_DIR, SEED
from ..data.preprocessing import ImageSplits, class_weights, load_image_splits, make_loaders
from .estimators import MODELOS, get_model
from .evaluate import append_result, evaluate_predictions, predict_loader
from .train import run_name, set_seed, train_model

CNN_CV_RESULTS_PATH = RESULTS_DIR / "cnn_cv_results.csv"


def cross_validate(
    modelo: str,
    root: Path = IMAGES_DIR,
    folds: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 5,
    batch_size: int = CNN_BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    pretrained: bool = True,
    freeze: bool = False,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """Treina e avalia ``folds`` vezes sobre treino + validação (o teste nunca entra).

    Returns:
        Uma linha por fold: ``fold``, ``acuracia``, ``f1_macro``, ``recall_macro``, ``precision_macro``, ``auc_ovr``,
        ``recall_<classe>`` de cada classe, ``melhor_epoca`` e ``tempo_s``.
    """
    splits = load_image_splits(root, save=False)
    desenvolvimento = pd.concat([splits.train, splits.val], ignore_index=True)  # o teste fica de fora
    divisao = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    opcoes = {"freeze_backbone": True} if freeze and modelo == "resnet18" else {}

    linhas = []
    for fold, (i_treino, i_val) in enumerate(divisao.split(desenvolvimento, desenvolvimento["rotulo"]), start=1):
        treino, validacao = desenvolvimento.iloc[i_treino], desenvolvimento.iloc[i_val]
        # o "test" do ImageSplits não é usado no treino; recebe a própria validação só para completar a estrutura
        loaders = make_loaders(ImageSplits(treino, validacao, validacao, splits.classes), batch_size, image_size)
        set_seed()
        inicio = time.perf_counter()
        resultado = train_model(get_model(modelo, len(splits.classes), pretrained=pretrained, **opcoes), loaders,
                                splits.classes, class_weights(treino, splits.classes), epochs=epochs, lr=lr,
                                patience=patience, device=device, verbose=False)
        y_true, y_prob = predict_loader(resultado.model, loaders["val"], device)
        m = evaluate_predictions(y_true, y_prob, splits.classes)
        por_classe = m["por_classe"].set_index("classe")["recall"]
        linhas.append({
            "fold": fold, "acuracia": m["acuracia"], "f1_macro": m["f1_macro"], "recall_macro": m["recall_macro"],
            "precision_macro": m["precision_macro"], "auc_ovr": m["auc_ovr"],
            **{f"recall_{classe}": float(valor) for classe, valor in por_classe.items()},
            "melhor_epoca": resultado.best_epoch, "tempo_s": time.perf_counter() - inicio,
        })
        if verbose:
            print(f"Fold {fold}/{folds}: F1 macro {m['f1_macro']:.3f} | recall macro {m['recall_macro']:.3f} | "
                  + " | ".join(f"recall {c} {v:.2f}" for c, v in por_classe.items()))
    return pd.DataFrame(linhas)


def summarize(resultados: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio-padrão de cada métrica entre os folds."""
    metricas = [c for c in resultados.columns if c not in ("fold", "melhor_epoca", "tempo_s")]
    return resultados[metricas].agg(["mean", "std"]).T.rename(columns={"mean": "media", "std": "desvio"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validação cruzada dos modelos de imagem (CNN-07)")
    parser.add_argument("--modelo", choices=list(MODELOS), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=CNN_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze", action="store_true", help="Congela o corpo do ResNet18")
    parser.add_argument("--no-pretrained", action="store_true", help="Não usa os pesos do ImageNet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--root", type=Path, default=IMAGES_DIR)
    args = parser.parse_args(argv)

    resultados = cross_validate(args.modelo, args.root, args.folds, args.epochs, args.lr, args.patience,
                                args.batch_size, args.image_size, not args.no_pretrained, args.freeze, args.device)
    execucao = run_name(args.modelo, args.image_size, args.freeze)
    for _, linha in resultados.iterrows():
        append_result({"modelo": execucao, "arquitetura": args.modelo, "image_size": args.image_size,
                       "freeze": args.freeze, **linha.to_dict()}, CNN_CV_RESULTS_PATH)
    resumo = summarize(resultados)
    print(f"\nValidação cruzada ({args.folds} folds) de {execucao}:")
    print(resumo.round(3).to_string())


if __name__ == "__main__":
    main()
