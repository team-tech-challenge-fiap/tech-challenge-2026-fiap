"""Pipeline de dados das imagens: split 70/15/15, transformações e DataLoaders (CNN-04).

É o equivalente do ``load_splits()`` do projeto tabular: **todo o time deve usar** ``load_image_splits()``,
com a mesma ``SEED``, para os modelos de imagem serem comparáveis. O teste é usado uma única vez, no final.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import (
    CNN_BATCH_SIZE,
    CNN_SPLIT_PATH,
    IMAGE_SIZE,
    IMAGES_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from .loader import scan_images

# Média e desvio do ImageNet: os modelos pré-treinados (transfer learning) esperam essa normalização.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

# Área da grade do traçado nas folhas do dataset (2213 x 1572 px), medida na EDA (CNN-03): a moldura está na mesma
# posição nas 491 imagens. Fora dela só há texto impresso (ID do paciente, data, hora e frequência cardíaca), que não
# é o ECG, pode virar atalho para o modelo e é dado pessoal. Guardada como fração da folha, vale em qualquer resolução.
FOLHA_ECG = (2213, 1572)
GRADE_ECG = (68, 283, 2176, 1517)  # (x0, y0, x1, y1) em pixels
RECORTE_GRADE = (
    GRADE_ECG[0] / FOLHA_ECG[0], GRADE_ECG[1] / FOLHA_ECG[1], GRADE_ECG[2] / FOLHA_ECG[0], GRADE_ECG[3] / FOLHA_ECG[1],
)


class RecorteGrade:
    """Recorta a área da grade do traçado, em fração da largura e da altura da imagem (primeiro passo do pipeline).

    Args:
        fracoes: ``(x0, y0, x1, y1)`` entre 0 e 1; o padrão é a grade das folhas deste dataset (``RECORTE_GRADE``).
    """

    def __init__(self, fracoes: tuple[float, float, float, float] = RECORTE_GRADE) -> None:
        x0, y0, x1, y1 = fracoes
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Frações de recorte inválidas: {fracoes}")
        self.fracoes = fracoes

    def __call__(self, img: Image.Image) -> Image.Image:
        largura, altura = img.size
        x0, y0, x1, y1 = self.fracoes
        return img.crop((round(x0 * largura), round(y0 * altura), round(x1 * largura), round(y1 * altura)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fracoes={tuple(round(f, 4) for f in self.fracoes)})"


class ImageSplits(NamedTuple):
    """Saída de ``load_image_splits``: uma tabela por partição e a lista de classes.

    Cada tabela tem ``caminho``, ``classe`` e ``rotulo`` (índice da classe em ``classes``).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    classes: list[str]


def load_image_splits(
    root: Path = IMAGES_DIR,
    seed: int = SEED,
    remove_duplicates: bool = True,
    save: bool = True,
    verbose: bool = False,
) -> ImageSplits:
    """Lê as imagens e divide em treino/validação/teste (70/15/15), estratificado por classe.

    As pastas ``Training/Testing`` do dataset original **são ignoradas**: o split é refeito aqui, com a
    mesma regra do projeto tabular, para termos uma validação e para o teste ser comparável.

    Args:
        root: pasta do dataset (``data/images`` por padrão).
        seed: semente do split (``config.SEED``).
        remove_duplicates: se True, remove duplicatas exatas (md5) **antes** de dividir, para a mesma
            imagem não cair no treino e no teste. Se a mesma imagem aparece com **rótulos diferentes**,
            todas as cópias saem (o rótulo é incerto); se o rótulo é o mesmo, fica uma cópia.
        save: se True, salva ``config.CNN_SPLIT_PATH`` (caminho relativo a ``root``, classe e partição).
        verbose: se True, imprime o tamanho de cada partição.

    Returns:
        ``ImageSplits(train, val, test, classes)``.
    """
    root = Path(root)
    imagens = scan_images(root)[["caminho", "classe"]]
    if remove_duplicates:
        imagens = _sem_duplicatas(imagens, verbose)

    classes = sorted(imagens["classe"].unique())
    imagens = imagens.assign(rotulo=imagens["classe"].map({c: i for i, c in enumerate(classes)}))

    resto, teste = train_test_split(imagens, test_size=TEST_SIZE, stratify=imagens["classe"], random_state=seed)
    treino, validacao = train_test_split(
        resto, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=resto["classe"], random_state=seed
    )
    treino, validacao, teste = (d.sort_values("caminho").reset_index(drop=True) for d in (treino, validacao, teste))

    if save:
        CNN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        salvo = pd.concat(
            [d.assign(particao=nome) for nome, d in (("train", treino), ("val", validacao), ("test", teste))]
        )
        salvo["caminho"] = [Path(c).relative_to(root).as_posix() for c in salvo["caminho"]]
        salvo[["caminho", "classe", "particao"]].sort_values("caminho").to_csv(CNN_SPLIT_PATH, index=False)

    if verbose:
        for nome, d in (("Treino", treino), ("Validação", validacao), ("Teste", teste)):
            print(f"{nome}: {len(d)} imagens ({len(d) / len(imagens):.0%}) | " + ", ".join(
                f"{c}: {n}" for c, n in d["classe"].value_counts().sort_index().items()))
    return ImageSplits(treino, validacao, teste, classes)


def _sem_duplicatas(imagens: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Remove duplicatas exatas (md5): mantém uma cópia se o rótulo é o mesmo e nenhuma se ele conflita."""
    imagens = imagens.assign(md5=[hashlib.md5(Path(c).read_bytes()).hexdigest() for c in imagens["caminho"]])
    conflitantes = imagens.groupby("md5")["classe"].transform("nunique") > 1
    limpas = imagens[~conflitantes].drop_duplicates("md5")
    if verbose:
        print(f"Duplicatas: {int(conflitantes.sum())} imagens com rótulos conflitantes removidas; "
              f"{len(imagens) - int(conflitantes.sum()) - len(limpas)} cópias iguais removidas")
    return limpas.drop(columns="md5").reset_index(drop=True)


def subsample(imagens: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Amostra estratificada de ``n`` imagens de uma partição, para testes rápidos (``--sample``)."""
    if n >= len(imagens):
        return imagens
    amostra, _ = train_test_split(imagens, train_size=n, stratify=imagens["classe"], random_state=seed)
    return amostra.sort_values("caminho").reset_index(drop=True)


def get_transforms(
    train: bool, size: int = IMAGE_SIZE, recorte: tuple[float, float, float, float] | None = RECORTE_GRADE
) -> transforms.Compose:
    """Transformações: recortar a grade, redimensionar, (aumento de dados só no treino), tensor e normalizar.

    O recorte (``RecorteGrade``) vem primeiro e vale para treino, validação, teste e predição, para o modelo ver
    sempre só o traçado. ``recorte=None`` desliga (ex.: imagens que não seguem o layout deste dataset).

    O aumento de dados cria pequenas variações das imagens de treino: deslocamento e rotação de poucos graus
    (papel do ECG escaneado ou fotografado levemente torto) e variação de brilho/contraste. **Não há giro
    horizontal nem vertical**: no ECG o eixo horizontal é o tempo e o vertical é a tensão elétrica, então inverter
    a imagem cria um traçado que não existe (a onda T e o QRS trocariam de ordem). Na validação e no teste
    **nunca** se aplica aumento: eles medem o modelo como ele será usado.
    """
    etapas: list = [RecorteGrade(recorte)] if recorte is not None else []
    etapas.append(transforms.Resize((size, size)))
    if train:
        etapas += [
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    etapas += [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET)]
    return transforms.Compose(etapas)


class ImageDataset(Dataset):
    """Lê as imagens de uma partição como tensores ``(3, lado, lado)`` e o rótulo (índice da classe)."""

    def __init__(self, imagens: pd.DataFrame, transform: transforms.Compose) -> None:
        self.caminhos = imagens["caminho"].tolist()
        self.rotulos = imagens["rotulo"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.caminhos[indice]) as img:
            return self.transform(img.convert("RGB")), self.rotulos[indice]  # cinza vira 3 canais iguais


def make_loaders(
    splits: ImageSplits,
    batch_size: int = CNN_BATCH_SIZE,
    size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
    recorte: tuple[float, float, float, float] | None = RECORTE_GRADE,
) -> dict[str, DataLoader]:
    """DataLoaders de treino (embaralhado e com aumento), validação e teste, todos com o mesmo ``recorte``.

    ``num_workers=0`` é o mais seguro no Windows; aumente se o carregamento for o gargalo.
    """
    gerador = torch.Generator().manual_seed(seed)  # mesma ordem de embaralhamento a cada execução
    return {
        nome: DataLoader(
            ImageDataset(dados, get_transforms(train=(nome == "train"), size=size, recorte=recorte)),
            batch_size=batch_size,
            shuffle=(nome == "train"),
            num_workers=num_workers,
            generator=gerador if nome == "train" else None,
        )
        for nome, dados in (("train", splits.train), ("val", splits.val), ("test", splits.test))
    }


def class_weights(treino: pd.DataFrame, classes: list[str]) -> torch.Tensor:
    """Peso de cada classe para a função de perda: ``n / (n_classes * contagem)``.

    Classes raras pesam mais; com classes balanceadas todos os pesos ficam perto de 1.
    """
    contagem = treino["classe"].value_counts().reindex(classes).to_numpy(dtype=float)
    return torch.tensor(contagem.sum() / (len(classes) * contagem), dtype=torch.float32)
