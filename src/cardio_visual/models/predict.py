"""Predição de imagens de ECG com um modelo treinado (CNN-09).

Uso:
    python -m cardio_visual.models.predict data/demo/ecg_1.jpg [outra.jpg ...]  # imagens quaisquer
    python -m cardio_visual.models.predict --demo             # as 4 imagens do teste cego (sem gabarito)
    python -m cardio_visual.models.predict --demo --gabarito  # e confere com data/demo_gabarito.csv

O resultado é **apoio à decisão**: a probabilidade de cada classe e a classe mais provável. Quem decide é o médico.
A imagem passa pelo mesmo pré-processamento da validação (recorte da grade, redimensionamento e normalização), sem
aumento de dados.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn

from ..config import IMAGE_EXTENSIONS, IMAGES_DIR, MODELS_DIR
from ..data.preprocessing import get_transforms
from .estimators import get_model

AVISO = "Apoio à decisão: o modelo não substitui a avaliação do médico, que tem a palavra final."
DEMO_DIR = IMAGES_DIR.parent / "demo"
GABARITO_PATH = IMAGES_DIR.parent / "demo_gabarito.csv"
# Modelo final, escolhido na comparação da validação (ver results/cnn_val_results.csv e README).
FINAL_CHECKPOINT = MODELS_DIR / "cnn_resnet18_448.pt"


def load_checkpoint(caminho: Path = FINAL_CHECKPOINT, device: str = "cpu") -> dict:
    """Carrega um checkpoint salvo por ``cardio_visual.models.train`` e recria o modelo, pronto para prever.

    Returns:
        ``{"model", "classes", "image_size", "modelo"}``; o modelo já vem em modo de avaliação.

    Raises:
        FileNotFoundError: se o checkpoint não existir (é gerado pelo treino; ver README).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {caminho}. "
                                "Treine antes com python -m cardio_visual.models.train.")
    dados = torch.load(caminho, map_location=device, weights_only=False)
    model = get_model(dados["modelo"], len(dados["classes"]), pretrained=False)
    model.load_state_dict(dados["state_dict"])
    return {"model": model.to(device).eval(), "classes": list(dados["classes"]), "image_size": int(dados["image_size"]),
            "modelo": dados["modelo"]}


def predict_image(checkpoint: dict, imagem: Path | Image.Image, device: str = "cpu") -> dict:
    """Classifica uma imagem.

    Args:
        checkpoint: saída de ``load_checkpoint``.
        imagem: caminho do arquivo ou imagem PIL.

    Returns:
        ``{"classe", "probabilidade", "probabilidades", "prob_alterado", "aviso"}``: a classe mais provável, a
        probabilidade dela, o dicionário com todas as classes, a probabilidade de **não** ser normal (soma das classes
        alteradas, a visão binária) e o aviso de apoio à decisão.
    """
    if isinstance(imagem, str | Path):
        with Image.open(imagem) as img:
            imagem = img.convert("RGB")
    entrada = get_transforms(train=False, size=checkpoint["image_size"])(imagem.convert("RGB")).unsqueeze(0)
    model: nn.Module = checkpoint["model"]
    with torch.no_grad():
        probabilidades = torch.softmax(model(entrada.to(device)), dim=1)[0].cpu().tolist()
    por_classe = dict(zip(checkpoint["classes"], probabilidades, strict=False))
    classe = max(por_classe, key=por_classe.get)
    return {"classe": classe, "probabilidade": por_classe[classe], "probabilidades": por_classe,
            "prob_alterado": 1.0 - por_classe.get("normal", 0.0), "aviso": AVISO}


def predict_files(checkpoint: dict, arquivos: list[Path], device: str = "cpu") -> pd.DataFrame:
    """Classifica vários arquivos; uma linha por imagem, com a probabilidade de cada classe."""
    linhas = []
    for arquivo in arquivos:
        r = predict_image(checkpoint, arquivo, device)
        linhas.append({"arquivo": Path(arquivo).name, "classe_prevista": r["classe"],
                       "probabilidade": r["probabilidade"], "prob_alterado": r["prob_alterado"],
                       **{f"p_{c}": p for c, p in r["probabilidades"].items()}})
    return pd.DataFrame(linhas)


def check_demo(previsoes: pd.DataFrame, gabarito: Path = GABARITO_PATH) -> pd.DataFrame:
    """Junta as previsões com o gabarito do teste cego e marca os acertos (``acertou``)."""
    resposta = pd.read_csv(gabarito)[["arquivo", "classe"]].rename(columns={"classe": "classe_real"})
    tabela = previsoes.merge(resposta, on="arquivo", how="left")
    tabela["acertou"] = tabela["classe_prevista"] == tabela["classe_real"]
    return tabela


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classifica imagens de ECG com o modelo treinado (CNN-09)")
    parser.add_argument("imagens", nargs="*", type=Path, help="Arquivos de imagem")
    parser.add_argument("--demo", action="store_true", help="Usa as imagens do teste cego (data/demo)")
    parser.add_argument("--gabarito", action="store_true", help="Com --demo, confere com data/demo_gabarito.csv")
    parser.add_argument("--checkpoint", type=Path, default=FINAL_CHECKPOINT)
    parser.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    args = parser.parse_args(argv)

    arquivos = list(args.imagens)
    if args.demo:
        arquivos += sorted(p for p in args.demo_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not arquivos:
        parser.error("informe imagens ou use --demo")

    checkpoint = load_checkpoint(args.checkpoint)
    previsoes = predict_files(checkpoint, arquivos)
    colunas = ["arquivo", "classe_prevista", "probabilidade", "prob_alterado"]
    if args.demo and args.gabarito:
        tabela = check_demo(previsoes, args.demo_dir.parent / "demo_gabarito.csv")
        print(tabela[colunas + ["classe_real", "acertou"]].round(3).to_string(index=False))
        print(f"\nAcertos: {int(tabela['acertou'].sum())} de {len(tabela)}")
    else:
        print(previsoes[colunas].round(3).to_string(index=False))
    print(f"\n{AVISO}")


if __name__ == "__main__":
    main()
