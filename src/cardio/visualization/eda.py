"""Gráficos da EDA, a partir de ``load_dataset()`` + ``clean()`` (``age_years`` em anos)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import SEED
from ..features.build import categorizar_pressao

descricao = {
    "id": "Identificador do paciente",
    "age_years": "Idade (anos)",
    "sex_male": "Sexo masculino",
    "height": "Altura (cm)",
    "weight": "Peso (kg)",
    "ap_hi": "Pressão arterial sistólica (mmHg)",
    "ap_lo": "Pressão arterial diastólica (mmHg)",
    "cholesterol": "Nível de colesterol",
    "gluc": "Nível de glicose",
    "smoke": "Tabagismo",
    "alco": "Consumo de álcool",
    "active": "Prática de atividade física",
    "cardio": "Presença de doença cardiovascular",
    "imc": "Índice de Massa Corporal (kg/m²)"
    }


# 1. Idade, tabagismo e doença cardiovascular
def plot_idade_tabagismo_cardio(df):
    
    fig, aux = plt.subplots(figsize=(12, 5))
   
    rng = np.random.default_rng(SEED)

    idade_jitter = df["age_years"] + rng.uniform(-0.3, 0.3, len(df))
    cardio_jitter = df["cardio"] + rng.uniform(-0.08, 0.08, len(df))
  
    scatter = aux.scatter (
    x = idade_jitter,
    y = cardio_jitter,
    c=df["smoke"],
    cmap="coolwarm", 
    s=2,
    alpha=0.05
    )

    aux.set_xlabel("Idade")
    aux.set_ylabel("Presença de doença cardiovascular")
    aux.set_title("Relação entre idade, tabagismo e presença de doença cardiovascular")
    plt.colorbar(scatter, ax=aux, label="Fumante (0 = Não, 1 = Sim)")

    plt.show()



# 2. Distribuição de doença cardiovascular
def plot_distribuicao_cardio(df):

    fig, aux = plt.subplots(figsize=(12, 5))

    count = df["cardio"].value_counts().sort_index()

    bars = aux.bar(
    x=count.index,
    height=count.values,
    color=["yellow", "green"]
   
)
    aux.set_xlabel("Diagnóstico")
    aux.set_ylabel("Quantidade de pacientes")
    aux.set_title("Distribuição de Doença Cardiovascular")
    aux.set_xticks(count.index, labels=["Sem doença cardiovascular", "Com doença cardiovascular"])
    plt.show()

# 3. Matriz de correlação
def plot_matriz_correlacao(df):
    
    fig, ax = plt.subplots(figsize=(10, 8))

    colunas = [
    "age_years",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
    ]

    correlacao = df[colunas].corr()
    heatmap = ax.imshow(correlacao, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(colunas)))
    ax.set_yticks(range(len(colunas)))
    ax.set_xticklabels([descricao[coluna] for coluna in colunas], rotation=45, ha="right")
    ax.set_yticklabels([descricao[coluna] for coluna in colunas])

    plt.title("Matriz de Correlação - Dados Cardiovasculares")
    plt.tight_layout()
    plt.show()

# 4. Distribuição de idade
def plot_distribuicao_idade(df):

    fig, ax = plt.subplots(figsize=(10, 8))

    hist = ax.hist(df["age_years"])

    ax.set_xlabel("Idade")
    ax.set_ylabel("Quantidade de Pacientes")
    ax.set_title("Distribuição de Idade dos Pacientes")

    plt.tight_layout()
    plt.show()

# 5. Distribuição das pressões sistólica e diastólica
def plot_distribuicao_pressao_arterial(df):
   
    colunas = ["ap_hi", "ap_lo"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    for ax, coluna in zip(axes, colunas):
        hist = ax.hist(df[coluna], bins=30, color="blue", alpha=0.7)
        ax.set_xlabel(descricao[coluna])
        ax.set_ylabel("Quantidade de Pacientes")
        ax.set_title(f"Distribuição de {descricao[coluna]}")

    plt.tight_layout()
    plt.show()

# 6. Pressão sistólica × doença cardiovascular
def plot_pressao_sistolica_cardio(df):

    sem_doenca = df.loc[df["cardio"] == 0, "ap_hi"]
    com_doenca = df.loc[df["cardio"] == 1, "ap_hi"]

    fig, ax = plt.subplots(figsize=(10, 8))

    box = ax.boxplot(
    [sem_doenca, com_doenca],
    tick_labels=["Sem doença", "Com doença"],
    showmeans=True
)

    ax.set_title("Pressão Sistólica × Doença Cardiovascular")
    ax.set_xlabel("Diagnóstico Cardiovascular")
    ax.set_ylabel("Pressão Sistólica (mmHg)")

    plt.tight_layout()
    plt.show()


# 7. Pressão diastólica × doença cardiovascular
def plot_pressao_diastolica_cardio(df):
    
    sem_doenca = df.loc[df["cardio"] == 0, "ap_lo"]
    com_doenca = df.loc[df["cardio"] == 1, "ap_lo"]

    fig, ax = plt.subplots(figsize=(10, 8))

    box = ax.boxplot(
    [sem_doenca, com_doenca],
    tick_labels=["Sem doença", "Com doença"],
    showmeans=True
    )

    ax.set_title("Pressão Diastólica × Doença Cardiovascular")
    ax.set_xlabel("Diagnóstico Cardiovascular")
    ax.set_ylabel("Pressão Diastólica (mmHg)")

    plt.tight_layout()
    plt.show()

# 8. Colesterol × doença cardiovascular
def plot_colesterol_cardio(df):

    tabela = pd.crosstab(df["cholesterol"], df["cardio"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(y=tabela.index, width=tabela[0], label="Sem doença", alpha=0.7)
    ax.barh(y=tabela.index, width=tabela[1], left=tabela[0], label="Com doença", alpha=0.7)

    ax.set_title("Colesterol × Doença Cardiovascular")
    ax.set_xlabel("Percentual de pacientes (%)")
    ax.set_ylabel("Nível de colesterol")

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Normal", "Acima do normal", "Muito acima"])

    ax.set_xlim(0, 100)
    ax.legend()

    for categoria in tabela.index:
        sem_doenca = tabela.loc[categoria, 0]
        com_doenca = tabela.loc[categoria, 1]

        ax.text(
            sem_doenca / 2, categoria,
            f"{sem_doenca:.1f}%",
            ha="center",
            va="center",
            color="white"
        )

        ax.text(
            sem_doenca + com_doenca / 2, categoria,
            f"{com_doenca:.1f}%",
            ha="center",
            va="center",
            color="white"
        )

    plt.tight_layout()
    plt.show()

# 9. Glicose × doença cardiovascular
def plot_glicose_cardio(df):

    tabela = pd.crosstab(df["gluc"], df["cardio"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(y=tabela.index, width=tabela[0], label="Sem doença", alpha=0.7)
    ax.barh(y=tabela.index, width=tabela[1], left=tabela[0], label="Com doença", alpha=0.7)

    ax.set_title("Glicose x Doença Cardiovascular")
    ax.set_xlabel("Percentual de pacientes (%)")
    ax.set_ylabel("Nível de glicose")

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Normal", "Acima do normal", "Muito acima"])

    ax.set_xlim(0, 100)
    ax.legend()

    for categoria in tabela.index:
        sem_doenca = tabela.loc[categoria, 0]
        com_doenca = tabela.loc[categoria, 1]

        ax.text(
            sem_doenca / 2, categoria,
            f"{sem_doenca:.1f}%",
            ha="center",
            va="center",
            color="white"
        )

        ax.text(
            sem_doenca + com_doenca / 2, categoria,
            f"{com_doenca:.1f}%",
            ha="center",
            va="center",
            color="white"
        )

    plt.tight_layout()
    plt.show()

# 10. Doença cardiovascular por faixa etária e tabagismo
def plot_faixa_etaria_tabagismo_cardio(df):
    
    df2 = df.copy()

    df2["faixa_etaria"] = pd.cut(
    df2["age_years"],
    bins=[20, 30, 40, 50, 60, 70, 80],
    right=False)

    tabela = pd.crosstab(
        [df2["faixa_etaria"], df2["smoke"]],
        df2["cardio"],
        normalize="index") * 100

    grafico = tabela[1].unstack("smoke")

    fig, ax = plt.subplots(figsize=(10, 8))
    grafico.plot(kind="bar", ax=ax, color=["green", "red"], alpha=0.7)

    ax.set_title("Doença Cardiovascular por Idade e Tabagismo")
    ax.set_xlabel("Faixa etária")
    ax.set_ylabel("Pacientes com doença cardiovascular (%)")
    ax.legend(["Não fumantes", "Fumantes"])


    #plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 11. Doença cardiovascular por categoria de pressão
def plot_categoria_pressao_cardio(df):

    df_pressao = categorizar_pressao(df)

    tabela = pd.crosstab(
    df_pressao["categoria_pressao"],
    df_pressao["cardio"],
    normalize="index"
) * 100

    ax = tabela[1].plot.bar(figsize=(10, 5))

    ax.set_title("Doença Cardiovascular por Categoria de Pressão")
    ax.set_xlabel("Categoria de pressão arterial")
    ax.set_ylabel("Pacientes com doença cardiovascular (%)")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_dados_invalidos(df):
    log = df.attrs["cleaning_log"]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.barh(
        log["regra"],
        log["linhas_removidas"],
        color="steelblue"
    )

    ax.invert_yaxis()

    ax.set_title("Registros removidos por inconsistência")
    ax.set_xlabel("Quantidade de registros removidos")
    ax.set_ylabel("Motivo da remoção")

    for i, valor in enumerate(log["linhas_removidas"]):
        ax.text(valor + 5, i, str(valor), va="center")

    ax.set_xlim(0, log["linhas_removidas"].max() * 1.15)

    plt.tight_layout()
    plt.show()