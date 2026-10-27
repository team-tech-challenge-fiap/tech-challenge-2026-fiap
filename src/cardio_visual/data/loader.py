"""Leitura das pastas de imagens e verificação de integridade (CNN-02).

Estrutura esperada: ``data/images/<classe>/<arquivo>.jpg``. Se o dataset já vier dividido
(``Training/`` e ``Testing/``), as pastas ficam como um nível acima da classe, e o nome delas é guardado
como "grupo original": ``data/images/Training/<classe>/<arquivo>.jpg``. A classe é sempre a pasta pai.

Datasets públicos de imagem médica costumam ter arquivos corrompidos, **duplicatas** e a mesma
imagem em partições diferentes. Isso infla a métrica no teste, então a verificação vem antes de treinar.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..config import IMAGE_EXTENSIONS, IMAGES_DIR

# Nome da pasta (minúsculo) -> grupo original. Outras pastas não são tratadas como grupo.
GRUPOS_ORIGINAIS = {
    "training": "treino", "train": "treino",
    "testing": "teste", "test": "teste",
    "validation": "validacao", "val": "validacao",
}
LADO_HASH_PERCEPTUAL = 16  # 16 x 16 = 256 bits


def scan_images(root: Path = IMAGES_DIR) -> pd.DataFrame:
    """Lista as imagens de ``root``, com a classe (pasta pai) e o grupo original (se houver).

    Args:
        root: pasta raiz do dataset (``config.IMAGES_DIR`` por padrão).

    Returns:
        DataFrame ordenado por caminho, com ``caminho`` (str), ``classe`` e ``grupo_original``
        (``"treino"``, ``"teste"``, ``"validacao"`` ou ``""``).

    Raises:
        FileNotFoundError: se a pasta não existir.
        ValueError: se não houver imagens ou houver menos de duas classes.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {root}")

    linhas = []
    for arquivo in sorted(root.rglob("*")):
        if not arquivo.is_file() or arquivo.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relativo = arquivo.relative_to(root)
        grupo = next(
            (GRUPOS_ORIGINAIS[p.lower()] for p in relativo.parts[:-2] if p.lower() in GRUPOS_ORIGINAIS), ""
        )
        linhas.append({"caminho": str(arquivo), "classe": arquivo.parent.name, "grupo_original": grupo})

    imagens = pd.DataFrame(linhas, columns=["caminho", "classe", "grupo_original"])
    if imagens.empty:
        raise ValueError(f"Nenhuma imagem ({', '.join(IMAGE_EXTENSIONS)}) encontrada em {root}")
    if imagens["classe"].nunique() < 2:
        raise ValueError(
            f"São necessárias ao menos 2 classes (pastas); encontrada: {imagens['classe'].unique().tolist()}"
        )
    return imagens


def inspect_images(imagens: pd.DataFrame) -> pd.DataFrame:
    """Abre cada imagem e acrescenta tamanho, modo de cor, hash exato e hash perceptual.

    - ``md5``: hash do arquivo; iguais = **duplicata exata**.
    - ``hash_perceptual``: hash de 16 x 16 pixels em tons de cinza (média); iguais mas com ``md5`` diferente =
      **quase-duplicata** (mesma imagem recomprimida ou redimensionada). É um sinal de alerta: revisar à mão.
    - ``erro``: texto do problema quando o arquivo não abre (``None`` se estiver íntegro).
    """
    resultados = []
    for caminho in imagens["caminho"]:
        dados = {"largura": None, "altura": None, "modo": None, "md5": None, "hash_perceptual": None, "erro": None}
        try:
            conteudo = Path(caminho).read_bytes()
            dados["md5"] = hashlib.md5(conteudo).hexdigest()
            with Image.open(caminho) as img:
                img.load()  # força a leitura completa: pega arquivos truncados
                dados["largura"], dados["altura"], dados["modo"] = img.width, img.height, img.mode
                dados["hash_perceptual"] = _hash_perceptual(img)
        except (OSError, UnidentifiedImageError, ValueError) as erro:
            dados["erro"] = f"{type(erro).__name__}: {erro}"
        resultados.append(dados)
    return pd.concat([imagens.reset_index(drop=True), pd.DataFrame(resultados)], axis=1)


def _hash_perceptual(img: Image.Image) -> str:
    pequena = img.convert("L").resize((LADO_HASH_PERCEPTUAL,) * 2, Image.Resampling.LANCZOS)
    pixels = np.asarray(pequena, dtype=float).ravel()
    bits = "".join("1" if p else "0" for p in pixels > pixels.mean())
    return f"{int(bits, 2):0{LADO_HASH_PERCEPTUAL ** 2 // 4}x}"


def find_duplicates(inspecionadas: pd.DataFrame, coluna: str = "md5") -> pd.DataFrame:
    """Imagens cujo hash (``coluna``) se repete, com ``id_duplicata`` para agrupar as cópias.

    Args:
        inspecionadas: saída de ``inspect_images``.
        coluna: ``"md5"`` (duplicata exata) ou ``"hash_perceptual"`` (quase-duplicata).
    """
    validas = inspecionadas[inspecionadas[coluna].notna()]
    repetidas = validas[validas.duplicated(coluna, keep=False)].copy()
    repetidas["id_duplicata"] = repetidas.groupby(coluna).ngroup()
    return repetidas.sort_values(["id_duplicata", "caminho"]).reset_index(drop=True)


def integrity_report(inspecionadas: pd.DataFrame) -> dict:
    """Resumo da verificação, pronto para o README e o relatório.

    Returns:
        Dicionário com: ``total``, ``por_classe`` (Series), ``corrompidas`` (DataFrame), ``modos`` e ``tamanhos``
        (contagens), ``duplicatas_exatas`` (nº de cópias extras), ``duplicatas_entre_classes`` (mesma imagem com
        rótulos diferentes: o pior caso), ``duplicatas_entre_grupos`` (mesma imagem em ``treino`` e ``teste`` etc.:
        vazamento) e ``quase_duplicatas`` (nº de imagens com o mesmo hash perceptual mas arquivo diferente).
    """
    ok = inspecionadas[inspecionadas["erro"].isna()]
    exatas = find_duplicates(inspecionadas, "md5")
    perceptuais = find_duplicates(inspecionadas, "hash_perceptual")
    quase = perceptuais[~perceptuais["md5"].isin(exatas["md5"])]

    def grupos_com_mais_de(coluna: str) -> int:
        if exatas.empty:
            return 0
        return int((exatas.groupby("id_duplicata")[coluna].nunique() > 1).sum())

    return {
        "total": len(inspecionadas),
        "por_classe": inspecionadas["classe"].value_counts(),
        "corrompidas": inspecionadas[inspecionadas["erro"].notna()][["caminho", "erro"]].reset_index(drop=True),
        "modos": ok["modo"].value_counts().to_dict(),
        "tamanhos": ok.groupby(["largura", "altura"]).size().sort_values(ascending=False).head(10).to_dict(),
        "duplicatas_exatas": int(len(exatas) - exatas["id_duplicata"].nunique()) if not exatas.empty else 0,
        "duplicatas_entre_classes": grupos_com_mais_de("classe"),
        "duplicatas_entre_grupos": grupos_com_mais_de("grupo_original"),
        "quase_duplicatas": int(len(quase)),
    }
