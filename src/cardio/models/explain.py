"""Interpretabilidade do modelo: feature importance (T24) e SHAP (T25).

A **feature importance** (T24) responde "quais variáveis o modelo mais usa?" de três jeitos: a importância
nativa das árvores, o odds ratio da Regressão Logística e a importância por permutação (quanto o modelo piora
quando uma variável é embaralhada), que vale para qualquer modelo.

O SHAP mostra **quanto cada variável empurrou o risco de um paciente para cima ou para baixo**, em
relação ao risco médio. Os valores estão na escala de log-odds do modelo: positivo aumenta a chance de
doença, negativo diminui, e a soma de todas as contribuições mais o valor-base dá a previsão do paciente.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos; evita abrir janela (testes, Docker, scripts)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter

# O shap importa o tqdm, que avisa quando o widget de progresso do Jupyter não está instalado (inofensivo).
warnings.filterwarnings("ignore", message="IProgress not found")
import shap  # noqa: E402
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline  # noqa: E402

from ..config import SEED  # noqa: E402
from .estimators.logistic import odds_ratios

# Nomes legíveis das colunas que saem do pré-processador (conjuntos A e B), para os gráficos.
NOMES_CLINICOS = {
    "age_years": "Idade (anos)",
    "sex_male": "Sexo masculino",
    "height": "Altura (cm)",
    "weight": "Peso (kg)",
    "ap_hi": "Pressão sistólica (mmHg)",
    "ap_lo": "Pressão diastólica (mmHg)",
    "cholesterol": "Colesterol (nível 1–3)",
    "gluc": "Glicose (nível 1–3)",
    "smoke": "Fumante",
    "alco": "Consome álcool",
    "active": "Fisicamente ativo",
    "imc": "IMC",
    "pressao_pulso": "Pressão de pulso",
    "pam": "Pressão arterial média",
}


# Incremento clínico em que o odds ratio de cada variável é lido (ex.: "a cada 10 mmHg"). Ordinais valem por
# nível e binárias por sim × não.
INCREMENTOS_CLINICOS = {
    "ap_hi": (10, "10 mmHg"),
    "ap_lo": (10, "10 mmHg"),
    "age_years": (5, "5 anos"),
    "weight": (10, "10 kg"),
    "height": (10, "10 cm"),
    "cholesterol": (1, "+1 nível"),
    "gluc": (1, "+1 nível"),
    "sex_male": (1, "homem × mulher"),
    "smoke": (1, "sim × não"),
    "alco": (1, "sim × não"),
    "active": (1, "sim × não"),
}


def nome_legivel(coluna: str) -> str:
    """Nome clínico de uma coluna (``ap_hi`` -> ``"Pressão sistólica (mmHg)"``); o que não está na tabela fica igual."""
    return NOMES_CLINICOS.get(coluna, coluna)


def importancia_nativa(pipeline: Pipeline) -> pd.DataFrame:
    """Importância nativa de um modelo de árvores (T24): Random Forest ou XGBoost.

    O Random Forest mede a redução média de impureza (Gini) que cada variável traz; o XGBoost, o ganho médio.
    Os valores são normalizados para somar 1. **Cuidado:** esse tipo de importância tende a inflar variáveis
    contínuas e correlacionadas com outras (ex.: pressão diastólica); compare com ``importancia_por_permutacao``.

    Args:
        pipeline: pipeline treinado (``"prep"`` + ``"model"``) cujo modelo tem ``feature_importances_``.

    Returns:
        DataFrame com ``variavel`` (nome legível) e ``importancia``, do maior para o menor.

    Raises:
        ValueError: se o modelo não tiver importância nativa (ex.: Regressão Logística, KNN, SVM).
    """
    modelo = pipeline[-1]
    if not hasattr(modelo, "feature_importances_"):
        raise ValueError(
            f"{type(modelo).__name__} não tem importância nativa; use importancia_por_permutacao()."
        )
    importancia = pd.Series(modelo.feature_importances_, index=pipeline[:-1].get_feature_names_out())
    importancia = importancia / importancia.sum()
    tabela = pd.DataFrame({
        "variavel": [nome_legivel(coluna) for coluna in importancia.index],
        "importancia": importancia.to_numpy(),
    })
    return tabela.sort_values("importancia", ascending=False, ignore_index=True)


def importancia_por_permutacao(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    seed: int = SEED,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Importância por permutação (T24): quanto o desempenho cai quando uma variável é embaralhada.

    Embaralhar quebra a relação da variável com o alvo; se o modelo piora muito, ele dependia dela. Vale para
    **qualquer** modelo. As colunas são embaralhadas **antes do pré-processador**, então cada variável original
    aparece uma vez (as categorias do conjunto B não ficam divididas em várias colunas one-hot).

    Variáveis correlacionadas (ex.: pressão sistólica e diastólica) dividem a importância entre si.

    Args:
        pipeline: pipeline treinado.
        X, y: dados **que o modelo não viu no treino** (a validação), no formato de ``load_splits()``.
        n_repeats: quantas vezes cada variável é embaralhada.
        scoring: métrica que cai; o padrão é a AUC, que não depende do limiar de decisão.
        seed: semente dos embaralhamentos.
        n_jobs: processos em paralelo.

    Returns:
        DataFrame com ``variavel`` (nome legível), ``queda_media`` e ``desvio`` entre as repetições,
        do maior para o menor.
    """
    resultado = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs
    )
    tabela = pd.DataFrame({
        "variavel": [nome_legivel(coluna) for coluna in X.columns],
        "queda_media": resultado.importances_mean,
        "desvio": resultado.importances_std,
    })
    return tabela.sort_values("queda_media", ascending=False, ignore_index=True)


def odds_ratios_clinicos(pipeline: Pipeline) -> pd.DataFrame:
    """Odds ratio da Regressão Logística em incrementos clínicos (T24).

    O odds ratio diz por quanto a **chance** de doença é multiplicada: 1,75 significa 75% a mais; 0,80, 20% a
    menos. Para as contínuas é lido por incremento clínico (ex.: a cada 10 mmHg de pressão sistólica); para as
    ordinais, por nível; para as binárias, sim × não. Vem de ``cardio.models.estimators.logistic.odds_ratios``
    (T15) e vale **mantendo as outras variáveis iguais**.

    Args:
        pipeline: pipeline da Regressão Logística já treinado (``build_model()`` + ``fit``).

    Returns:
        DataFrame com ``variavel`` (nome legível), ``incremento``, ``odds_ratio`` (no incremento clínico) e
        ``odds_ratio_por_desvio`` (por desvio-padrão, comparável entre as variáveis); ordenado pelo tamanho
        do efeito por desvio-padrão, do maior para o menor.
    """
    tabela = odds_ratios(pipeline)
    linhas = []
    for _, linha in tabela.iterrows():
        incremento, descricao = INCREMENTOS_CLINICOS.get(linha["variavel"], (1, "por unidade"))
        linhas.append({
            "variavel": nome_legivel(linha["variavel"]),
            "incremento": descricao,
            "odds_ratio": float(linha["odds_ratio_por_unidade"] ** incremento),
            "odds_ratio_por_desvio": float(linha["odds_ratio"]),
        })
    return pd.DataFrame(linhas)  # odds_ratios() já vem ordenado pelo efeito por desvio-padrão


def shap_importancia(explicacao) -> pd.Series:
    """Importância pelo SHAP: contribuição absoluta média de cada variável (T25), indexada pelo nome legível."""
    return pd.Series(np.abs(explicacao.values).mean(axis=0), index=explicacao.feature_names)


def comparar_rankings(importancias: dict[str, pd.Series]) -> pd.DataFrame:
    """Posição de cada variável em cada método (1 = mais importante), da mais para a menos importante na média.

    Args:
        importancias: um ``Series`` por método (valor maior = mais importante), indexado pelo nome da variável.

    Returns:
        DataFrame com uma linha por variável e uma coluna por método.
    """
    ranking = pd.DataFrame(importancias).rank(ascending=False, method="min").astype(int)
    return ranking.loc[ranking.mean(axis=1).sort_values(kind="stable").index]


def plot_importancias(
    tabelas: dict[str, pd.DataFrame],
    coluna: str,
    path: Path,
    title: str,
    xlabel: str,
    percentual: bool = False,
    erro: str | None = None,
) -> Path:
    """Barras horizontais, um painel por modelo, cada um do mais para o menos importante (T24).

    Args:
        tabelas: um DataFrame por painel, com ``variavel`` e a ``coluna`` de valores.
        coluna: coluna com a importância.
        path: arquivo ``.png`` de saída.
        title: título da figura.
        xlabel: rótulo do eixo horizontal.
        percentual: se True, mostra ``coluna`` × 100 com o símbolo %.
        erro: coluna com o desvio, desenhado como barra de erro.
    """
    fator = 100 if percentual else 1
    _, eixos = plt.subplots(1, len(tabelas), figsize=(5.4 * len(tabelas), 5.2), sharex=True)
    for ax, (nome, tabela) in zip(np.atleast_1d(eixos), tabelas.items(), strict=True):
        dados = tabela.sort_values(coluna)  # no barh o último valor fica no topo
        valores = dados[coluna].to_numpy() * fator
        barras = ax.barh(dados["variavel"], valores, xerr=None if erro is None else dados[erro] * fator,
                         color="tab:blue", error_kw={"ecolor": "gray", "lw": 1})
        ax.bar_label(barras, labels=[_formatar(v, percentual) for v in valores], padding=3, fontsize=8)
        ax.set(title=nome, xlabel=xlabel)
        ax.margins(x=0.18)
    return _salvar(path, title)


def plot_odds_ratios(tabela: pd.DataFrame, path: Path, title: str) -> Path:
    """Odds ratio por incremento clínico: vermelho aumenta a chance de doença; azul diminui (T24)."""
    dados = tabela.iloc[::-1]  # o mais forte no topo
    rotulos = [f"{variavel}\n(por {incremento})" for variavel, incremento in zip(dados["variavel"], dados["incremento"])]
    _, ax = plt.subplots(figsize=(8.5, 6.5))
    cores = ["tab:red" if valor > 1 else "tab:blue" for valor in dados["odds_ratio"]]
    ax.scatter(dados["odds_ratio"], rotulos, color=cores, s=60, zorder=3)
    ax.hlines(rotulos, 1, dados["odds_ratio"], color=cores, lw=2, zorder=2)
    for rotulo, valor in zip(rotulos, dados["odds_ratio"]):
        ax.annotate(_formatar(valor, False, casas=2), (valor, rotulo), xytext=(8 if valor > 1 else -8, 0),
                    textcoords="offset points", ha="left" if valor > 1 else "right", va="center", fontsize=9)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set(xscale="log", xlabel="Odds ratio (escala logarítmica; 1 = sem efeito)")
    ax.set_xticks([0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _formatar(v, False, casas=1)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.12)
    return _salvar(path, title)


def plot_ranking(ranking: pd.DataFrame, path: Path, title: str) -> Path:
    """Mapa de calor das posições (1 = mais importante) de cada variável em cada método (T24)."""
    n = len(ranking)
    _, ax = plt.subplots(figsize=(1.15 * ranking.shape[1] + 3.5, 0.5 * n + 2))
    ax.imshow(ranking.to_numpy(), cmap="Blues_r", vmin=1, vmax=n, aspect="auto")
    ax.set_xticks(range(ranking.shape[1]), ranking.columns, rotation=30, ha="right")
    ax.set_yticks(range(n), ranking.index)
    for i in range(n):
        for j in range(ranking.shape[1]):
            posicao = ranking.iat[i, j]
            ax.text(j, i, posicao, ha="center", va="center", color="white" if posicao <= n / 2 else "black")
    ax.grid(False)
    return _salvar(path, title)


def _formatar(valor: float, percentual: bool, casas: int | None = None) -> str:
    casas = (1 if percentual else 3) if casas is None else casas
    return f"{valor:.{casas}f}{'%' if percentual else ''}".replace(".", ",")


def shap_values(pipeline: Pipeline, X: pd.DataFrame, sample_size: int | None = 1500, seed: int = SEED):
    """Calcula os valores SHAP de um pipeline de árvores (``"prep"`` + ``"model"``) (T25).

    O SHAP é aplicado aos dados **já transformados** pelo pré-processador (``pipeline[:-1]``), que é o que
    o modelo de fato recebe; os nomes das colunas vêm de ``get_feature_names_out()``.

    Args:
        pipeline: pipeline treinado cujo último passo é um modelo de árvores (XGBoost ou Random Forest).
        X: features no formato de ``load_splits()`` (antes do pré-processador).
        sample_size: tamanho da amostra (padrão 1.500; ``None`` usa todas as linhas). Calcular em todos os
            pacientes é lento e não muda a interpretação.
        seed: semente da amostragem.

    Returns:
        ``(explicacao, X_amostra)``: um ``shap.Explanation`` com os valores e os dados transformados (com
        nomes legíveis em ``explicacao.feature_names``) e as linhas originais da amostra, no mesmo índice.
    """
    if sample_size is not None and sample_size < len(X):
        X = X.sample(sample_size, random_state=seed)
    X_transformado = pipeline[:-1].transform(X)
    explicacao = shap.TreeExplainer(pipeline[-1])(X_transformado)
    explicacao.feature_names = [NOMES_CLINICOS.get(nome, nome) for nome in X_transformado.columns]
    return explicacao, X


def shap_por_faixa(explicacao, coluna: str, bins) -> pd.DataFrame:
    """Contribuição SHAP média de uma variável por faixa de valores (para ler o dependence plot).

    Args:
        explicacao: saída de ``shap_values()``.
        coluna: nome legível da variável (ex.: ``"Pressão sistólica (mmHg)"``).
        bins: limites das faixas, como em ``pd.cut``.

    Returns:
        DataFrame com ``faixa``, ``pacientes`` e ``shap_medio`` (positivo = aumenta o risco).
    """
    j = explicacao.feature_names.index(coluna)
    tabela = pd.DataFrame({"valor": explicacao.data[:, j], "shap": explicacao.values[:, j]})
    tabela["faixa"] = pd.cut(tabela["valor"], bins=bins, right=False)
    resumo = tabela.groupby("faixa", observed=True)["shap"].agg(pacientes="size", shap_medio="mean")
    return resumo.reset_index()


def plot_shap_summary(explicacao, path: Path, title: str) -> Path:
    """Summary plot (beeswarm): cada ponto é um paciente; a cor é o valor da variável."""
    shap.plots.beeswarm(explicacao, max_display=len(explicacao.feature_names), show=False)
    return _salvar(path, title)


def plot_shap_dependence(explicacao, coluna: str, path: Path, title: str) -> Path:
    """Dependence plot: como a contribuição de uma variável muda conforme o valor dela."""
    j = explicacao.feature_names.index(coluna)
    shap.plots.scatter(explicacao[:, j], show=False)
    plt.axhline(0, color="gray", lw=1, ls="--")
    return _salvar(path, title)


def plot_shap_waterfall(explicacao, posicao: int, path: Path, title: str) -> Path:
    """Waterfall plot de um paciente: como cada variável levou do risco médio até o risco dele."""
    shap.plots.waterfall(explicacao[posicao], max_display=12, show=False)
    return _salvar(path, title)


def _salvar(path: Path, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def probabilidade(log_odds) -> np.ndarray:
    """Converte a escala do SHAP (log-odds) em probabilidade: 0 -> 50%, +1 -> 73%, -1 -> 27%."""
    return 1 / (1 + np.exp(-np.asarray(log_odds)))
