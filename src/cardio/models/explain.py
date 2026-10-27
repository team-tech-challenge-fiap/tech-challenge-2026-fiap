"""Interpretabilidade do modelo: feature importance (T24) e SHAP (T25).

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

# O shap importa o tqdm, que avisa quando o widget de progresso do Jupyter não está instalado (inofensivo).
warnings.filterwarnings("ignore", message="IProgress not found")
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ..config import SEED  # noqa: E402

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


def feature_importance(model, features, target):
    """TODO (Luis, T24): importância das variáveis (ex.: permutation_importance do sklearn)."""
    raise NotImplementedError


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
