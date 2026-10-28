"""Pipeline do extra de imagens (CNN-10): é o comando padrão do ``Dockerfile.visual``.

**Padrão: caminho rápido** (~1 min na CPU, sem treinar), para quem só quer ver o modelo funcionando. Usa o modelo final
que já está no repositório (``models/cnn_resnet18_448.pt``):

1. ``data.preparar``: extrai o zip do Kaggle para ``data/images/<classe>/`` e reserva o teste cego
   (se ainda não foi feito);
2. teste cego: classifica as 4 imagens de ``data/demo`` e confere com o gabarito;
3. avaliação no teste com o modelo final, só **exibida**: os arquivos versionados de ``results/`` não são regravados.

**Com ``--treinar``: caminho completo** (~25 min na CPU), como o ``cardio.pipeline`` do projeto tabular:
preparar -> treinar o modelo final (``MODELO_FINAL`` em ``IMAGE_SIZE_FINAL`` px) -> avaliação única no teste,
gravando o checkpoint e os resultados.

Uso: ``python -m cardio_visual.pipeline [--zip data/archive.zip] [--treinar [--epochs 20]]``.
"""

import argparse
import tempfile
import time
from pathlib import Path

from .config import KAGGLE_ZIP_PATH
from .data.preparar import prepare_images
from .models import final_model, predict, train
from .models.predict import FINAL_CHECKPOINT

# Configuração escolhida na comparação da validação (ver README); o checkpoint final é FINAL_CHECKPOINT.
MODELO_FINAL = "resnet18"
IMAGE_SIZE_FINAL = 448
EPOCHS_FINAL = 20


def fast_path() -> dict:
    """Teste cego e avaliação no teste com o modelo final do repositório, sem treinar e sem regravar ``results/``."""
    print("\n=== 2/3 · Teste cego (4 imagens nunca vistas pelo modelo) ===")
    predict.main(["--demo", "--gabarito", "--checkpoint", str(FINAL_CHECKPOINT)])
    print(f"\n=== 3/3 · Avaliação no teste com o modelo final ({FINAL_CHECKPOINT.name}) ===")
    with tempfile.TemporaryDirectory() as temporaria:
        pasta = Path(temporaria)
        r = final_model.evaluate_final(FINAL_CHECKPOINT, figures_dir=pasta, results_path=pasta / "teste.csv",
                                       por_classe_path=pasta / "por_classe.csv")
    print(r["por_classe"].round(3).to_string(index=False))
    print(f"\n4 classes: acurácia {r['acuracia']:.3f} | F1 macro {r['f1_macro']:.3f} "
          f"(IC 95%: {r['f1_macro_ic'][0]:.3f} a {r['f1_macro_ic'][1]:.3f})")
    print(f"Binário (normal × alterado): recall {r['recall_alterado']:.3f} | "
          f"especificidade {r['especificidade']:.3f}")
    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline do extra de imagens de ECG")
    parser.add_argument("--zip", type=Path, default=KAGGLE_ZIP_PATH, help="Zip baixado do Kaggle")
    parser.add_argument("--treinar", action="store_true",
                        help="Caminho completo: treina o modelo final antes de avaliar (~25 min na CPU)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_FINAL, help="Épocas do treino (só com --treinar)")
    args = parser.parse_args(argv)

    inicio = time.perf_counter()
    print("=== 1/3 · Preparação das imagens ===")
    prepare_images(args.zip)
    if args.treinar:
        print(f"\n=== 2/3 · Treino do modelo final ({MODELO_FINAL}, {IMAGE_SIZE_FINAL} px) ===")
        train.main(["--modelo", MODELO_FINAL, "--image-size", str(IMAGE_SIZE_FINAL), "--epochs", str(args.epochs)])
        print("\n=== 3/3 · Avaliação única no teste ===")
        final_model.main(["--checkpoint", str(FINAL_CHECKPOINT)])
    else:
        fast_path()
    print(f"\nPipeline concluído em {time.perf_counter() - inicio:.1f} s.")


if __name__ == "__main__":
    main()
