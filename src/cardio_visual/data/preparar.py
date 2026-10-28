"""Prepara as imagens do dataset do Kaggle a partir do zip baixado (CNN-02).

Uso: ``python -m cardio_visual.data.preparar [--zip data/archive.zip] [--destino data/images] [--sobrescrever]``.

O zip do Kaggle vem com as pastas ``train/`` e ``test/``, mas elas **não são confiáveis**: na verificação deste dataset,
todas as imagens de ``test/`` são cópias de imagens de ``train/``, e o próprio ``train/`` repete imagens (algumas
até 8 vezes). Por isso a preparação:

1. junta ``train/`` e ``test/`` numa **pasta só por classe** (``data/images/<classe>/``), com nomes curtos de classe;
2. guarda **uma cópia** de cada imagem (comparando o conteúdo pelo hash MD5); se a mesma imagem aparece com classes
   diferentes, ela é descartada, porque o rótulo é incerto;
3. grava ``origem.csv`` com, para cada imagem, os arquivos originais do zip de onde ela veio;
4. **reserva 1 imagem de cada classe para um teste cego** (professor e apresentação): elas saem de ``data/images`` e vão
   para ``data/demo/ecg_1.jpg`` ... ``ecg_4.jpg``, com nomes que não revelam a classe; a resposta fica em
   ``data/demo_gabarito.csv``. Como saem antes do split, o modelo nunca treina, valida nem testa com elas.

A divisão em treino, validação e teste é feita depois, por nós (``cardio_visual.data.preprocessing.load_image_splits``).
"""

import argparse
import hashlib
import random
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from ..config import IMAGE_EXTENSIONS, IMAGES_DIR, KAGGLE_URL, KAGGLE_ZIP_PATH, SEED

# Trecho do nome da pasta no Kaggle (minúsculo) -> nome curto da classe. A ordem importa: "history of mi" vem antes.
CLASSES_KAGGLE = (
    ("history of mi", "historico_infarto"),
    ("myocardial infarction", "infarto"),
    ("abnormal heartbeat", "batimento_anormal"),
    ("normal", "normal"),
)
ORIGEM = "origem.csv"
DEMO = "demo"  # pasta irmã de data/images com as imagens do teste cego
GABARITO = "demo_gabarito.csv"  # resposta do teste cego, fora da pasta entregue a quem testa


def class_name(pasta: str) -> str:
    """Nome curto da classe a partir do nome da pasta do Kaggle.

    Ex.: "Normal Person ECG Images (284x12=3408)" -> "normal".

    Pastas desconhecidas viram um nome simples: minúsculas, sem acentos de pontuação e com ``_``.
    """
    minusculo = pasta.lower()
    for trecho, nome in CLASSES_KAGGLE:
        if trecho in minusculo:
            return nome
    return re.sub(r"[^a-z0-9]+", "_", minusculo).strip("_") or "sem_nome"


def _caminho_seguro(nome: str) -> PurePosixPath | None:
    """Caminho do arquivo dentro do zip, ou None se ele tentar sair da pasta (absoluto ou com "..")."""
    caminho = PurePosixPath(nome.replace("\\", "/"))
    if caminho.is_absolute() or ".." in caminho.parts or re.match(r"^[a-zA-Z]:", nome):
        return None
    return caminho


def reserve_demo(destino: Path = IMAGES_DIR, seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    """Separa 1 imagem de cada classe para um teste cego e devolve o gabarito.

    A escolha é **sorteada** (``random.Random(seed)``, uma por pasta de classe, em ordem alfabética de classe e de
    arquivo), não escolhida a dedo, e a ordem ``ecg_1`` ... ``ecg_N`` é embaralhada, para o número não indicar a classe.
    As imagens são **movidas** para ``destino.parent / "demo"``: saem do treino, da validação e do teste.
    Se o gabarito já existe, não faz nada (a reserva é feita uma vez só).

    Returns:
        Gabarito: ``arquivo`` (nome em ``demo/``), ``classe``, ``md5`` e ``origem`` (caminho em ``destino``).
    """
    destino = Path(destino)
    demo, gabarito = destino.parent / DEMO, destino.parent / GABARITO
    if gabarito.exists():
        return pd.read_csv(gabarito)

    rng = random.Random(seed)
    escolhidas = []
    for classe in sorted(p.name for p in destino.iterdir() if p.is_dir()):
        arquivos = sorted(p for p in (destino / classe).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if arquivos:
            escolhidas.append((classe, rng.choice(arquivos)))
    rng.shuffle(escolhidas)

    demo.mkdir(parents=True, exist_ok=True)
    linhas = []
    for i, (classe, origem) in enumerate(escolhidas, start=1):
        alvo = demo / f"ecg_{i}{origem.suffix.lower()}"
        linhas.append({"arquivo": alvo.name, "classe": classe, "md5": hashlib.md5(origem.read_bytes()).hexdigest(),
                       "origem": f"{classe}/{origem.name}"})
        shutil.move(str(origem), alvo)
    tabela = pd.DataFrame(linhas, columns=["arquivo", "classe", "md5", "origem"])
    tabela.to_csv(gabarito, index=False)
    if verbose:
        print(f"Teste cego: {len(tabela)} imagens (1 por classe, sorteadas) separadas em {demo}; "
              f"gabarito em {gabarito}")
    return tabela


def _desfazer_demo(destino: Path) -> None:
    """Apaga a reserva anterior (imagens de demo e gabarito), para ``--sobrescrever`` refazer tudo do zero."""
    demo, gabarito = destino.parent / DEMO, destino.parent / GABARITO
    if gabarito.exists():
        for arquivo in pd.read_csv(gabarito)["arquivo"]:
            (demo / arquivo).unlink(missing_ok=True)
        gabarito.unlink()


def prepare_images(
    zip_path: Path = KAGGLE_ZIP_PATH,
    destino: Path = IMAGES_DIR,
    sobrescrever: bool = False,
    verbose: bool = True,
    reservar_demo: bool = True,
) -> pd.DataFrame:
    """Extrai as imagens do zip para ``destino/<classe>/``, uma cópia de cada, e devolve o resumo por classe.

    Args:
        zip_path: zip baixado do Kaggle.
        destino: pasta das imagens (``data/images`` por padrão).
        sobrescrever: se False e a pasta já foi preparada (existe ``origem.csv``), não refaz a extração. Se True, apaga
            só as imagens listadas no ``origem.csv`` e no gabarito anteriores e prepara de novo.
        verbose: se True, imprime o resumo.
        reservar_demo: se True, separa 1 imagem de cada classe para o teste cego (``reserve_demo``).

    Returns:
        DataFrame com ``classe``, ``arquivos_no_zip`` e ``imagens_unicas`` (a última linha é o total).

    Raises:
        FileNotFoundError: se o zip não existir (o README explica como baixar).
    """
    zip_path, destino = Path(zip_path), Path(destino)
    origem_csv = destino / ORIGEM
    if origem_csv.exists() and not sobrescrever:
        if verbose:
            print(f"As imagens já estão preparadas em {destino} (use --sobrescrever para refazer).")
        if reservar_demo:
            reserve_demo(destino, verbose=verbose)
        return _resumo(pd.read_csv(origem_csv))
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Zip do dataset não encontrado: {zip_path}. Baixe em {KAGGLE_URL} (ver README, seção do extra).")
    if origem_csv.exists():
        for arquivo in pd.read_csv(origem_csv)["arquivo"]:
            (destino / arquivo).unlink(missing_ok=True)
        origem_csv.unlink()
        _desfazer_demo(destino)

    registros, ignorados = [], 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            caminho = _caminho_seguro(info.filename)
            imagem_de_classe = (caminho is not None and caminho.suffix.lower() in IMAGE_EXTENSIONS
                                and len(caminho.parts) >= 2)
            if info.is_dir() or not imagem_de_classe:
                ignorados += not info.is_dir()
                continue
            conteudo = z.read(info)
            registros.append({"md5": hashlib.md5(conteudo).hexdigest(), "classe": class_name(caminho.parent.name),
                              "original": str(caminho), "extensao": caminho.suffix.lower(), "conteudo": conteudo})

    tabela = pd.DataFrame(registros)
    if tabela.empty:
        raise ValueError(f"Nenhuma imagem ({', '.join(IMAGE_EXTENSIONS)}) dentro de pastas de classe em {zip_path}")
    classes_por_md5 = tabela.groupby("md5")["classe"].nunique()
    conflitantes = set(classes_por_md5[classes_por_md5 > 1].index)

    origem = []
    for md5, grupo in tabela[~tabela["md5"].isin(conflitantes)].groupby("md5", sort=True):
        primeira = grupo.iloc[0]
        arquivo = PurePosixPath(primeira["classe"]) / f"{md5[:16]}{primeira['extensao']}"
        (destino / arquivo).parent.mkdir(parents=True, exist_ok=True)
        (destino / arquivo).write_bytes(primeira["conteudo"])
        origem.append({"arquivo": str(arquivo), "classe": primeira["classe"], "md5": md5, "copias": len(grupo),
                       "arquivos_originais": "; ".join(sorted(grupo["original"]))})

    origem = pd.DataFrame(origem).sort_values(["classe", "arquivo"]).reset_index(drop=True)
    destino.mkdir(parents=True, exist_ok=True)
    origem.to_csv(origem_csv, index=False)
    resumo = _resumo(origem)
    if verbose:
        print(f"{len(tabela)} imagens no zip -> {len(origem)} únicas em {destino}")
        n_conflitantes = int(tabela["md5"].isin(conflitantes).sum())
        print(f"Descartadas: {len(tabela) - len(origem) - n_conflitantes} cópias repetidas, "
              f"{n_conflitantes} imagens com classes conflitantes; "
              f"{ignorados} arquivos que não são imagem de classe")
        print(resumo.to_string(index=False))
    if reservar_demo:
        reserve_demo(destino, verbose=verbose)
    return resumo


def _resumo(origem: pd.DataFrame) -> pd.DataFrame:
    por_classe = origem.groupby("classe").agg(arquivos_no_zip=("copias", "sum"), imagens_unicas=("md5", "size"))
    por_classe = por_classe.reset_index()
    total = pd.DataFrame([{"classe": "total", "arquivos_no_zip": por_classe["arquivos_no_zip"].sum(),
                           "imagens_unicas": por_classe["imagens_unicas"].sum()}])
    return pd.concat([por_classe, total], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepara as imagens de ECG a partir do zip do Kaggle (CNN-02)")
    parser.add_argument("--zip", type=Path, default=KAGGLE_ZIP_PATH, help="Zip baixado do Kaggle")
    parser.add_argument("--destino", type=Path, default=IMAGES_DIR, help="Pasta de saída (uma subpasta por classe)")
    parser.add_argument("--sobrescrever", action="store_true", help="Refaz a preparação se ela já existir")
    parser.add_argument("--sem-demo", action="store_true", help="Não reserva as imagens do teste cego")
    args = parser.parse_args(argv)
    prepare_images(args.zip, args.destino, args.sobrescrever, reservar_demo=not args.sem_demo)


if __name__ == "__main__":
    main()
