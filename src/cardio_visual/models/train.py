"""Treino dos modelos de imagem com early stopping (CNN-07).

Uso: ``python -m cardio_visual.models.train --modelo resnet18 --epochs 15 [--sample 200] [--freeze]``.

O modelo é escolhido **só com a validação**: a cada época mede-se o F1 macro na validação (num empate, vale a menor
perda na validação), e o treino para quando ele deixa de melhorar por ``--patience`` épocas, restaurando o melhor
estado. O teste não é usado aqui.
"""

import argparse
import copy
import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import CNN_BATCH_SIZE, CNN_RESULTS_PATH, FIGURES_DIR, IMAGE_SIZE, IMAGES_DIR, MODELS_DIR, SEED
from ..data.preprocessing import class_weights, load_image_splits, make_loaders, subsample
from .estimators import MODELOS, get_model
from .evaluate import append_result, evaluate_model, evaluate_predictions, predict_loader


class TrainResult(NamedTuple):
    """Saída de ``train_model``.

    Attributes:
        model: o modelo, com os pesos da **melhor** época (pelo F1 macro na validação).
        history: uma linha por época, com ``epoca``, ``loss_treino``, ``loss_val``, ``acuracia_val``,
            ``f1_macro_val`` e ``recall_macro_val``.
        best_epoch: época (a partir de 1) com o melhor F1 macro na validação.
        train_seconds: tempo total de treino.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    train_seconds: float


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes (Python, NumPy e PyTorch) para o treino ser reprodutível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    classes: list[str],
    pesos: torch.Tensor | None = None,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
    seed: int = SEED,
    verbose: bool = True,
) -> TrainResult:
    """Treina com Adam e perda ponderada por classe, parando cedo pela validação.

    Args:
        model: modelo de ``cardio_visual.models.estimators`` ainda sem treino.
        loaders: saída de ``make_loaders`` (usa ``"train"`` e ``"val"``; nunca ``"test"``).
        classes: nomes das classes, na ordem dos índices.
        pesos: pesos de classe da função de perda (``class_weights``); ``None`` = todos iguais.
        epochs: máximo de épocas.
        lr: taxa de aprendizado. Só os parâmetros treináveis entram no otimizador (o corpo congelado fica de fora).
        patience: épocas sem melhora do F1 macro na validação antes de parar.
        device: ``"cpu"`` ou ``"cuda"``.
        seed: semente do treino.
        verbose: se True, imprime uma linha por época.
    """
    set_seed(seed)
    gerador = getattr(loaders["train"], "generator", None)
    if gerador is not None:  # o embaralhamento tem gerador próprio: reinicia, senão a ordem muda a cada chamada
        gerador.manual_seed(seed)
    model = model.to(device)
    perda = nn.CrossEntropyLoss(weight=None if pesos is None else pesos.to(device))
    otimizador = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    historico, melhor_f1, melhor_loss, melhor_estado, melhor_epoca, sem_melhora = [], -1.0, float("inf"), None, 0, 0
    inicio = time.perf_counter()
    for epoca in range(1, epochs + 1):
        model.train()
        perdas = []
        for imagens, rotulos in loaders["train"]:
            otimizador.zero_grad()
            valor = perda(model(imagens.to(device)), rotulos.to(device))
            valor.backward()
            otimizador.step()
            perdas.append(valor.item())

        y_true, y_prob = predict_loader(model, loaders["val"], device)
        m = evaluate_predictions(y_true, y_prob, classes)
        loss_val = float(-np.log(np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean())
        historico.append({
            "epoca": epoca, "loss_treino": float(np.mean(perdas)), "loss_val": loss_val,
            "acuracia_val": m["acuracia"], "f1_macro_val": m["f1_macro"], "recall_macro_val": m["recall_macro"],
        })
        if verbose:
            print(f"Época {epoca:02d} | loss treino {np.mean(perdas):.3f} | loss val {loss_val:.3f} | "
                  f"F1 macro val {m['f1_macro']:.3f} | recall macro val {m['recall_macro']:.3f}")

        # Melhora = F1 maior; com F1 empatado (comum em validações pequenas, que saturam cedo), vale a perda menor.
        empate = abs(m["f1_macro"] - melhor_f1) <= 1e-4
        if m["f1_macro"] > melhor_f1 + 1e-4 or (empate and loss_val < melhor_loss - 1e-4):
            melhor_f1, melhor_loss, melhor_epoca, sem_melhora = m["f1_macro"], loss_val, epoca, 0
            melhor_estado = copy.deepcopy(model.state_dict())
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                if verbose:
                    print(f"Parada antecipada: sem melhora há {patience} épocas (melhor: época {melhor_epoca}).")
                break

    model.load_state_dict(melhor_estado)
    return TrainResult(model, pd.DataFrame(historico), melhor_epoca, time.perf_counter() - inicio)


def run_name(modelo: str, image_size: int, freeze: bool = False) -> str:
    """Nome de uma execução de treino (ex.: ``resnet18_448``): identifica checkpoint, figuras e linha de resultados."""
    return f"{modelo}_{image_size}" + ("_freeze" if freeze else "")


def save_checkpoint(model: nn.Module, classes: list[str], nome: str, image_size: int, path: Path | None = None) -> Path:
    """Salva os pesos junto com o que a predição precisa saber: nome do modelo, classes e tamanho da imagem."""
    path = Path(path) if path is not None else MODELS_DIR / f"cnn_{run_name(nome, image_size)}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "modelo": nome, "classes": classes, "image_size": image_size}, path)
    return path


def _dispositivo(escolha: str) -> str:
    if escolha == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return escolha


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treino de modelos de imagem (CNN-07)")
    parser.add_argument("--modelo", choices=list(MODELOS), required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=CNN_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--sample", type=int, default=None, help="Nº de imagens de treino, para testes rápidos")
    parser.add_argument("--freeze", action="store_true", help="Congela o corpo do ResNet18 (treina só a última camada)")
    parser.add_argument("--no-pretrained", action="store_true", help="Não baixa/usa os pesos do ImageNet")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--root", type=Path, default=IMAGES_DIR, help="Pasta das imagens (uma subpasta por classe)")
    args = parser.parse_args(argv)

    device = _dispositivo(args.device)
    set_seed()  # antes de criar o modelo: os pesos iniciais (camadas novas) também precisam ser reprodutíveis
    splits = load_image_splits(args.root, verbose=True)
    treino = splits.train if args.sample is None else subsample(splits.train, args.sample)
    loaders = make_loaders(splits._replace(train=treino), batch_size=args.batch_size, size=args.image_size)

    opcoes = {"freeze_backbone": True} if args.freeze and args.modelo == "resnet18" else {}
    modelo = get_model(args.modelo, len(splits.classes), pretrained=not args.no_pretrained, **opcoes)
    print(f"Modelo: {args.modelo} | classes: {splits.classes} | treino: {len(treino)} imagens | dispositivo: {device}")

    resultado = train_model(modelo, loaders, splits.classes, class_weights(treino, splits.classes),
                            epochs=args.epochs, lr=args.lr, patience=args.patience, device=device)
    inicio = time.perf_counter()
    execucao = run_name(args.modelo, args.image_size, args.freeze)
    m = evaluate_model(resultado.model, loaders["val"], splits.classes, name=execucao, split="val",
                       figures_dir=FIGURES_DIR, device=device)
    tempo_predicao = time.perf_counter() - inicio

    caminho = save_checkpoint(resultado.model, splits.classes, args.modelo, args.image_size,
                              MODELS_DIR / f"cnn_{execucao}.pt")
    append_result({
        "modelo": execucao, "arquitetura": args.modelo, "acuracia": m["acuracia"], "f1_macro": m["f1_macro"],
        "recall_macro": m["recall_macro"],
        "precision_macro": m["precision_macro"], "auc_ovr": m["auc_ovr"],
        "recall_macro_ic_baixo": m["recall_macro_ic"][0], "recall_macro_ic_alto": m["recall_macro_ic"][1],
        "f1_macro_ic_baixo": m["f1_macro_ic"][0], "f1_macro_ic_alto": m["f1_macro_ic"][1],
        "epocas": len(resultado.history), "melhor_epoca": resultado.best_epoch,
        "tempo_treino_s": resultado.train_seconds, "tempo_predicao_val_s": tempo_predicao,
        "n_treino": len(treino), "n_val": len(splits.val),
        "pretrained": not args.no_pretrained, "freeze": args.freeze, "image_size": args.image_size,
    }, CNN_RESULTS_PATH)

    print(m["por_classe"].round(3).to_string(index=False))
    print(f"F1 macro na validação: {m['f1_macro']:.3f} (IC 95%: {m['f1_macro_ic'][0]:.3f} a {m['f1_macro_ic'][1]:.3f})")
    print(f"Checkpoint salvo em: {caminho}")


if __name__ == "__main__":
    main()
