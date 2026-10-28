# Cardio IA — Tech Challenge FIAP Fase 1

Sistema de apoio à triagem que estima o risco de doença cardiovascular a partir de dados clínicos simples.
Projeto da Fase 1 do Tech Challenge da Pós-Tech FIAP.

> **Aviso:** o modelo é apenas um **apoio à decisão médica**. Ele ajuda a priorizar quem examinar primeiro,
> mas não dá diagnóstico: a palavra final é sempre do médico.

## O problema e o objetivo

As doenças cardiovasculares são a principal causa de morte no mundo, e muitos casos só são descobertos quando
já há sintomas graves. Em postos de saúde e triagens com muitos pacientes, nem todos conseguem fazer exames
mais completos logo de início.

O objetivo do projeto é verificar se dados que qualquer atendimento coleta (idade, sexo, altura, peso, pressão
arterial, colesterol, glicose e hábitos de vida) permitem apontar quem tem mais chance de ter doença
cardiovascular, para que essas pessoas sejam examinadas primeiro.

Para isso, o time explorou e limpou os dados, comparou cinco algoritmos de classificação (Regressão Logística,
KNN, Random Forest, XGBoost e SVM) sempre com a mesma divisão em treino, validação e teste, ajustou o limiar de
decisão para priorizar a detecção de doentes e explicou as decisões do modelo final com SHAP. O resultado está
resumido em [Resultados](#resultados). Como complemento opcional, o projeto também classifica imagens de
eletrocardiograma com redes neurais (seção [Extra](#extra-diagnóstico-por-imagem-de-ecg-cnn)).

## Integrantes e links

| Integrante | GitHub |
|---|---|
| Guilherme Saar Santos | [@GuilhermeSaarSantos](https://github.com/GuilhermeSaarSantos) |
| Wesley Berto | [@WesleyBert](https://github.com/WesleyBert) |
| Michel Zarif | [@MichelZarif](https://github.com/MichelZarif) |
| Luis Felipe Laudacio | [@luislaudacio](https://github.com/luislaudacio) |
| Sara Felix | [@sarafelix25](https://github.com/sarafelix25) |

- **Vídeo de demonstração:** _link a publicar (T36)_
- **Relatório técnico:** [`results/relatorio_tecnico.md`](results/relatorio_tecnico.md) (o PDF final sai na T35)
- **Repositório:** <https://github.com/team-tech-challenge-fiap/tech-challenge-2026-fiap>

## Guia rápido

Com o [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e **aberto**, na pasta do projeto:

```bash
docker build -t cardio-classifier .
```

```powershell
docker run --rm -v "${PWD}/results:/app/results" -v "${PWD}/models:/app/models" cardio-classifier
```

O primeiro comando monta a imagem e o segundo roda o pipeline final: treina o modelo escolhido e o avalia uma
única vez no teste. Os detalhes, a versão para Linux/macOS e a opção sem Docker estão nas seções
[Executando com Docker](#executando-com-docker) e [Executando sem Docker](#executando-sem-docker).

| O que | Comando (Docker) | Tempo aproximado |
|---|---|---|
| Montar a imagem | `docker build -t cardio-classifier .` | 5 a 6 min (só na primeira vez) |
| Pipeline final: treino do modelo final + teste | `docker run ... cardio-classifier` (acima) | 20 s |
| Testes automatizados (137) | `docker compose run --rm tests` | 1 min |
| Treinar um modelo e avaliar na validação | `docker compose run --rm train` | 15 s a 1 min por modelo |
| Busca de hiperparâmetros de um modelo (`--tune`) | ver [Executando sem Docker](#executando-sem-docker) | 4 a 8 min por modelo |
| Notebook final (`00_projeto_final.ipynb`) | `docker compose up jupyter` | 15 s para rodar inteiro |

O notebook final **carrega o modelo salvo pelo pipeline final** (`models/final_model.joblib`, que não vai para o
Git), então rode o pipeline antes de abrir o notebook. Tempos medidos num computador com 8 núcleos.

## Dataset

O dataset é o **Cardiovascular Disease Dataset** do Kaggle:
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>.
São 70.000 pacientes e 13 colunas, num CSV separado por `;`.

O arquivo **já vem no repositório**, em `data/raw/cardio_train.csv`, e não é preciso baixar nada. Se ele faltar,
baixe o `cardio_train.csv` na página do Kaggle (exige login) e salve exatamente nesse caminho. O código confere
as colunas ao carregar e avisa se alguma estiver faltando.

| Coluna | Significado | Observação |
|---|---|---|
| `id` | identificador | **não usar como feature** |
| `age` | idade | **em dias** (convertida para anos) |
| `gender` | sexo | 1 = mulher, 2 = homem (as médias de altura confirmam: 161 vs. 170 cm) |
| `height`, `weight` | altura (cm) e peso (kg) | |
| `ap_hi`, `ap_lo` | pressão sistólica e diastólica | contém valores impossíveis (tratados na limpeza, T05 e T08) |
| `cholesterol`, `gluc` | colesterol e glicose | 1 = normal, 2 = acima do normal, 3 = muito acima |
| `smoke`, `alco`, `active` | fuma, bebe, faz atividade física | autodeclarados (0/1) |
| `cardio` | **alvo** | 1 = doença cardiovascular |

## Resultados

Depois da limpeza ficam 68.509 pacientes, divididos uma única vez em treino (70%), validação (15%) e teste (15%).
Os modelos foram comparados na **validação**, com o corte padrão de 0,5 (conjunto A, as 11 variáveis originais):

| Modelo | AUC | Recall | Precision | F1 | Acurácia |
|---|---|---|---|---|---|
| **XGBoost (modelo final)** | **0,792** | 0,679 | 0,745 | 0,711 | 0,726 |
| SVM | 0,792 | 0,693 | 0,741 | 0,716 | 0,728 |
| Random Forest | 0,791 | 0,673 | 0,752 | 0,711 | 0,729 |
| KNN | 0,785 | 0,660 | 0,753 | 0,704 | 0,725 |
| Regressão Logística | 0,783 | 0,671 | 0,750 | 0,708 | 0,727 |
| Referência (sorteio) | 0,508 | 0,494 | 0,504 | 0,499 | 0,509 |

Fontes: `results/comparison_table.csv` (T23) e, para o SVM, `results/val_results.csv` (T19, treinado depois da
comparação da T23; empata com o XGBoost, mas não o supera). Os cinco algoritmos ficam no mesmo patamar de AUC:
o limite está nas variáveis disponíveis, não no algoritmo. O conjunto B (com as variáveis derivadas) mudou a AUC
em menos de 0,01 em todos os modelos.

**Modelo final no teste**, avaliado uma única vez (XGBoost, conjunto A, limiar de decisão ajustado para 0,36 na
validação para encontrar pelo menos 80% dos doentes):

| AUC | Recall | Precision | F1 | Especificidade | Acurácia |
|---|---|---|---|---|---|
| 0,810 | 0,831 | 0,669 | 0,742 | 0,598 | 0,713 |

Na prática, **a cada 1.000 pacientes**, o modelo deixa de sinalizar cerca de 83 doentes e encaminha cerca de 203
pessoas saudáveis para um exame de confirmação. O corte em 0,36, em vez de 0,5, troca alarmes falsos por menos
doentes perdidos, porque numa triagem o falso negativo é o erro mais grave (fonte: `results/test_results.csv`).

<p align="center">
  <img src="results/figures/roc_A.png" alt="Curvas ROC dos cinco modelos na validação" width="48%">
  <img src="results/figures/t25_shap_summary.png" alt="SHAP: contribuição de cada variável no modelo final" width="48%">
</p>

À esquerda, as curvas ROC dos cinco modelos praticamente se sobrepõem. À direita, o SHAP mostra que a pressão
sistólica, a idade e o colesterol são o que mais pesa na decisão do modelo final, o que concorda com os fatores
de risco conhecidos na medicina. Lembrando: o modelo é **apoio à decisão**, nunca diagnóstico. Detalhes e
limitações estão no [relatório técnico](results/relatorio_tecnico.md).

## Estrutura

O projeto separa **código reutilizável** (`src/cardio/`, um pacote Python importado por todos) de **análises**
(`notebooks/`, que só chamam funções do pacote) e de **resultados gerados** (`results/` e `models/`):

- `data/`: o dataset original, que nunca é alterado.
- `src/cardio/`: limpeza, divisão dos dados, variáveis, modelos, avaliação e o pipeline final.
- `src/cardio_visual/`: o extra de imagens de ECG, com dependências e Docker próprios (seção Extra).
- `notebooks/`: um notebook por tarefa, numerado; o `00_projeto_final.ipynb` conta a história inteira.
- `results/`: tabelas (`.csv`), figuras (`figures/`) e o relatório técnico.
- `models/`: modelos treinados (`.joblib`), gerados pelos comandos de treino.
- `tests/`: testes automatizados (`pytest`), que conferem dados, divisão, modelos e avaliação.

```text
data/
  raw/cardio_train.csv           # dataset original
  processed/                     # dados preparados (gerados)
  images/<classe>/               # extra: ECGs por classe, gerados por cardio_visual.data.preparar (fora do Git)
  demo/                          # extra: 4 ECGs do teste cego (fora do Git; gabarito em data/demo_gabarito.csv)
notebooks/
  00_projeto_final.ipynb         # T29: notebook de ponta a ponta (rode o pipeline final antes)
  01_eda_visao_geral.ipynb       # T03: visão geral, alvo e qualidade dos dados
  02_eda_demografia_imc.ipynb    # T04: idade, sexo, altura, peso e IMC
  03_eda_pressao_arterial.ipynb  # T05: pressão arterial
  04_eda_exames_habitos.ipynb    # T06: colesterol, glicose e hábitos de vida
  05_feature_engineering.ipynb   # T10: variáveis derivadas e conjuntos A e B
  06_modelo_regressao_logistica.ipynb  # T15: regressão logística (baseline) e odds ratio
  07_ajuste_limiar.ipynb         # T22: ajuste do limiar de decisão (recall >= 0,80) e impacto por 1.000 pacientes
  08_eda_correlacao.ipynb        # T07: correlação, multicolinearidade, checagem de vazamento e relações não lineares
  09_shap.ipynb                 # T25: interpretação do modelo final com SHAP (summary, dependence e waterfall)
  10_analise_erros.ipynb         # T26: análise de erros e desempenho por subgrupo (modelo final, limiar da T22)
  11_feature_importance.ipynb    # T24: importância nativa, odds ratio, permutação e comparação com o conhecimento clínico
  cnn_01_eda_imagens.ipynb       # extra (CNN-03): EDA das imagens de ECG
  cnn_02_interpretacao.ipynb     # extra (CNN-08): erros e Grad-CAM do modelo final
  cnn_03_projeto_imagens.ipynb   # extra (CNN-11): demonstração de ponta a ponta e teste cego
  12_resultados_relatorio.ipynb  # T33: números e figuras das seções 6 e 7 do relatório (resultados e interpretação)
src/cardio/
  config.py                      # SEED, caminhos e contrato do dataset
  pipeline.py                    # pipeline final (T27): modelo final + avaliação no teste (comando padrão do Docker)
  data/
    loader.py                    # leitura + validação das colunas, conversão de idade e sexo
    preprocessing.py             # limpeza (clean), split 70/15/15 (load_splits), get_cv e build_preprocessor (T11)
  features/
    build.py                     # T10: add_features, categorizar_pressao e conjuntos A e B
  models/
    estimators/                  # um arquivo por modelo: build_model() + PARAM_GRID (T13, T15–T19)
    tuning.py                    # T14: tune() = busca de hiperparâmetros com CV no treino
    train.py                     # CLI de treino: --modelo, --features, --sample (T13)
    evaluate.py                  # T20: evaluate(), matriz de confusão, summary_table() e baseline
    predict.py                   # predição com modelo salvo
    explain.py                   # SHAP (T25) e feature importance (T24)
  visualization/
    eda.py                       # gráficos da EDA
    plots.py                     # TODO: gráficos para o relatório
  api/
    main.py                      # API FastAPI (/health; /predict TODO)
src/cardio_visual/                # extra: diagnóstico por imagem de ECG (CNN), mesma organização do cardio
  config.py                      # caminhos e parâmetros do extra (reaproveita SEED e split do cardio)
  pipeline.py                    # preparar -> treinar o modelo final -> avaliar no teste (padrão do Docker do extra)
  data/
    preparar.py                  # zip do Kaggle -> data/images/<classe>/, sem cópias, e reserva do teste cego
    loader.py                    # leitura das pastas e verificação de integridade
    preprocessing.py             # recorte da grade, split 70/15/15, transformações e DataLoaders
  models/
    estimators/                  # um arquivo por modelo: build_model() (cnn_simples, resnet18)
    train.py                     # CLI de treino com early stopping
    cv.py                        # CLI de validação cruzada (5 folds)
    evaluate.py                  # métricas por classe, intervalos de confiança e resultados
    explain.py                   # Grad-CAM e análise de erros
    predict.py                   # CLI de predição e teste cego
    final_model.py               # CLI da avaliação única no teste
  visualization/
    plots.py                     # matriz de confusão e curvas ROC
app/
  streamlit_app.py               # TODO: interface de demonstração (vídeo)
models/                          # modelos gerados (.joblib) e o modelo final do extra (cnn_resnet18_448.pt)
results/
  figures/                       # gráficos gerados (.png)
  relatorio_tecnico.md           # relatório técnico (base do PDF final)
  comparison_table.csv           # T23: comparação dos modelos na validação
  test_results.csv               # T23: avaliação única do modelo final no teste
tests/                           # testes automatizados
Dockerfile · docker-compose.yml · requirements.txt · pyproject.toml
Dockerfile.visual · docker-compose.visual.yml · requirements-cnn.txt   # Docker e dependências do extra
```

## Onde fica cada coisa citada nas issues

As issues foram escritas com uma estrutura simplificada. Os caminhos reais são:

| Na issue | No projeto |
|---|---|
| `data/cardio_train.csv` | `data/raw/cardio_train.csv` |
| `notebooks/05_eda_correlacao.ipynb` (T07) | `notebooks/08_eda_correlacao.ipynb` (o 05 já é a T10) |
| `notebooks/08_analise_erros.ipynb` (T26) | `notebooks/10_analise_erros.ipynb` (o 08 é a T07 e o 09 a T25) |
| `notebooks/06_feature_importance.ipynb` (T24) | `notebooks/11_feature_importance.ipynb` (o 06 já é a T15) |
| `src/config.py` | `src/cardio/config.py` |
| `src/data.py` (T02) | `src/cardio/data/loader.py` |
| `src/preprocessing.py` (T08, T09, T11) | `src/cardio/data/preprocessing.py` |
| `src/features.py` (T10) | `src/cardio/features/build.py` |
| `src/tuning.py` (T14) | `src/cardio/models/tuning.py` |
| `src/train.py` (T13) | `src/cardio/models/train.py` |
| `src/evaluate.py` (T20) | `src/cardio/models/evaluate.py` |
| `src/models/<modelo>.py` (T15–T19) | `src/cardio/models/estimators/<modelo>.py` |
| `results/figures/` | `results/figures/` |

Imports: `from cardio.config import SEED, DATA_PATH` (o pacote é `cardio`, dentro de `src/`).

## Executando com Docker

Pré-requisito: [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e **aberto**.
A imagem usa `python:3.11-slim` e instala as versões fixadas do `requirements.txt`. O dataset já vai dentro da imagem.

### Opção 1: `docker run` (avaliação / quem só quer rodar o projeto)

```bash
docker build -t cardio-classifier .
```

**Rodar o pipeline final** (comando padrão): treina o modelo escolhido na T23 (XGBoost, conjunto A) com
treino + validação, aplica o limiar da T22 (0,36) e avalia **uma única vez** no teste. Para os resultados
aparecerem na sua máquina, monte as pastas `results/` e `models/`:

```powershell
# Windows (PowerShell)
docker run --rm -v "${PWD}/results:/app/results" -v "${PWD}/models:/app/models" cardio-classifier
```

```bash
# Linux / macOS
docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/models:/app/models" cardio-classifier
```

Saída: `models/final_model.joblib` (modelo + limiar) e `results/test_results.csv` (accuracy, AUC, recall,
precision, F1, especificidade e impacto a cada 1.000 pacientes). O pipeline leva cerca de 20 segundos, quase
tudo para carregar as bibliotecas (o build da imagem, só na primeira vez, leva de 5 a 6 minutos).

**Treinar outro modelo** e avaliá-lo na validação (acrescente o comando no fim):

```bash
docker run --rm -v "${PWD}/results:/app/results" -v "${PWD}/models:/app/models" cardio-classifier \
  python -m cardio.models.train --modelo random_forest --features B
```

**Abrir o Jupyter** e acessar http://localhost:8888 (a pasta do projeto é montada, então o que for salvo nos
notebooks fica na sua máquina). O `-e MPLBACKEND=...` faz os gráficos aparecerem no notebook:

```powershell
docker run --rm -p 8888:8888 -v "${PWD}:/app" -e MPLBACKEND=module://matplotlib_inline.backend_inline cardio-classifier jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=''
```

Para ver o projeto de ponta a ponta, abra `notebooks/00_projeto_final.ipynb` e use **Run > Run All Cells**
(cerca de 15 s; precisa do pipeline final rodado antes).

**Rodar os testes** (cerca de 1 minuto; 137 passam e 8 do extra de imagens são pulados, porque a imagem não tem
o PyTorch):

```bash
docker run --rm cardio-classifier pytest -q
docker run --rm cardio-classifier pytest tests/test_data.py -q   # só qualidade dos dados e vazamento (T12)
```

### Opção 2: `docker compose` (dia a dia do time)

Monta a pasta inteira do projeto como volume, então alterações no código valem sem reconstruir a imagem.

```bash
docker compose build                  # constrói a imagem (repetir só quando o requirements.txt mudar)
docker compose up jupyter             # Jupyter em http://localhost:8888 (Ctrl + C para parar)
docker compose run --rm pipeline      # pipeline final: modelo final + avaliação no teste
docker compose run --rm train         # treina o Random Forest (conjunto A) na validação -> models/random_forest_A.joblib
docker compose run --rm tests         # roda os testes
docker compose up api app             # API (http://localhost:8000/docs) e demo (http://localhost:8501)
```

### Problemas comuns

| Erro | Solução |
|---|---|
| `cannot connect to the Docker daemon` | O Docker Desktop não está aberto |
| `port is already allocated` | O Jupyter já está rodando em outro terminal: feche-o ou rode `docker compose down` |
| Resultados não aparecem em `results/` | Faltou o `-v` no `docker run` (sem volume, o que é gerado fica dentro do container) |

## Executando sem Docker

Pré-requisito: **Python 3.11**, a mesma versão da imagem Docker e a testada com as versões fixadas do
`requirements.txt`. A instalação leva cerca de 6 minutos.

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\activate
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

Depois, nos dois sistemas:

```bash
python -m pip install -r requirements.txt   # bibliotecas nas versões fixadas
python -m pip install -e . --no-deps        # o pacote cardio (src/), sem reinstalar as bibliotecas

python -m cardio.pipeline                   # pipeline final: modelo final + avaliação no teste (~20 s)
pytest                                      # todos os testes (~1 min; os do extra são pulados sem o PyTorch)
pytest tests/test_data.py                   # só qualidade dos dados e vazamento (T12)
python -m cardio.models.train --modelo xgb --features A          # um modelo na validação (15 s a 1 min)
python -m cardio.models.train --modelo svm --features A --tune   # com busca de hiperparâmetros (4 a 8 min)
jupyter lab                                 # abre o Jupyter; rode o notebooks/00_projeto_final.ipynb (~15 s)
```

No Windows, se a instalação falhar com `No such file or directory` num caminho muito longo, crie o `.venv` numa
pasta mais curta (ex.: `C:\venvs\cardio`): o Windows limita caminhos a 260 caracteres.

`tests/test_data.py` (T12) garante que `id` e `cardio` nunca entram nas features, que a limpeza não deixa
valores fora das faixas, que treino, validação e teste não têm pacientes em comum, que o split é
reprodutível e estratificado e que o pré-processador é ajustado só no treino.

O treino usa `load_splits()` (70/15/15), avalia na **validação**, salva `models/<modelo>_<A|B>.joblib` e
acrescenta uma linha em `results/val_results.csv`. `--modelo` é o nome de um arquivo em
`src/cardio/models/estimators/`; use `--sample 5000` para testes rápidos.

## Extra: diagnóstico por imagem de ECG (CNN)

Complemento opcional do desafio: classificação de imagens de eletrocardiograma (ECG) com redes neurais convolucionais
(PyTorch). Fica no pacote **`src/cardio_visual/`**, separado do `cardio` (projeto tabular), com dependências próprias.

### Dataset: como baixar

As imagens **não ficam no Git** (são ~300 MB zipadas). Dataset: [Cardiovascular ECG Images (Kaggle)](https://www.kaggle.com/datasets/jayaprakashpondy/ecgimages/data).

1. Entre no Kaggle com a sua conta (o download exige login).
2. Na página do dataset, clique em **Download** e depois em **Download dataset as zip** (~300 MB, 905 MB descompactado).
3. Salve o arquivo como `data/archive.zip` (é o nome que o Kaggle já usa).
   Alternativa pela API do Kaggle (exige o token `kaggle.json`):
   `kaggle datasets download -d jayaprakashpondy/ecgimages -p data` (gera `data/ecgimages.zip`; use `--zip` com esse caminho no passo 4).
4. Prepare as imagens: `python -m cardio_visual.data.preparar` (ou `--zip <caminho do zip>`).
   A preparação também **reserva 1 imagem de cada classe para um teste cego** (professor e apresentação): sorteadas com
   `SEED = 42`, saem do treino e vão para `data/demo/ecg_1.jpg` ... `ecg_4.jpg`, com nomes que não revelam a classe.
   A resposta fica em `data/demo_gabarito.csv` (fora da pasta entregue a quem testa).

O zip vem com as pastas `train/` e `test/`, mas **elas não são usadas como estão**: todas as imagens vão para **uma pasta só por
classe** em `data/images/<classe>/`, sem cópias repetidas, e a divisão em treino, validação e teste é **feita por nós**
(70/15/15 estratificado, `SEED = 42`), como no projeto tabular. O motivo está na seção de qualidade do dataset.

### Qualidade do dataset (verificada em 27/10/2026)

`python -m cardio_visual.data.preparar` mostra este resumo. O zip tem 1.376 arquivos, mas só **491 imagens diferentes**:

| Classe (pasta no Kaggle) | Nome curto | Arquivos no zip | Imagens únicas |
|---|---|---|---|
| ECG Images of Myocardial Infarction Patients | `infarto` | 351 | **30** |
| ECG Images of Patient that have History of MI | `historico_infarto` | 284 | 86 |
| ECG Images of Patient that have abnormal heartbeat | `batimento_anormal` | 345 | 233 |
| Normal Person ECG Images | `normal` | 396 | 142 |

- **Todas as 448 imagens de `test/` são cópias de imagens de `train/`**, com a mesma classe. Avaliar no `test/` original
  mediria o modelo em imagens que ele já viu no treino.
- O próprio `train/` repete imagens: cada ECG de infarto aparece até 8 vezes, e os de histórico de infarto e normal, 2 vezes.
- Nenhum arquivo corrompido; todas as imagens são RGB de 2213 × 1572 pixels (ECG de 12 derivações); nenhuma quase-duplicata.
- O dataset não tem descrição nem licença no Kaggle ("Unknown").

Consequência: a classe `infarto` tem só 30 imagens, então a validação e o teste terão poucas imagens dela e a métrica dessa
classe é pouco confiável. `data/images/origem.csv` registra de quais arquivos do zip veio cada imagem.

A EDA das imagens (balanceamento, integridade, layout, intensidade, resolução e a decisão entre 4 classes ou binário) está em
`notebooks/cnn_01_eda_imagens.ipynb`. As imagens trazem o ID do paciente, a data e a frequência cardíaca impressos: as figuras
cobrem esses campos, e o recomendado é treinar só com a área da grade do traçado.

### Instalação

`python -m pip install -e ".[cnn]"` (PyTorch CPU; para GPU, instale o PyTorch com CUDA antes).

### Docker do extra

O extra tem **imagem Docker própria** (`Dockerfile.visual`, com PyTorch CPU) e um compose separado, para não pesar a imagem
do projeto tabular. As imagens não entram na imagem Docker: o zip fica em `data/archive.zip` e a pasta do projeto é montada
como volume.

```bash
docker compose -f docker-compose.visual.yml build
docker compose -f docker-compose.visual.yml run --rm preparar           # extrai data/archive.zip e reserva o teste cego
docker compose -f docker-compose.visual.yml run --rm demo               # classifica as 4 imagens do teste cego
docker compose -f docker-compose.visual.yml run --rm gabarito           # idem, conferindo com o gabarito
docker compose -f docker-compose.visual.yml run --rm treino             # treina o modelo final e avalia na validação
docker compose -f docker-compose.visual.yml run --rm validacao-cruzada  # 5 folds sobre treino + validação
docker compose -f docker-compose.visual.yml run --rm final              # avaliação única no teste
docker compose -f docker-compose.visual.yml run --rm rapido             # rápido (~1 min): teste cego + avaliação, sem treinar
docker compose -f docker-compose.visual.yml run --rm completo           # completo (~25 min): preparar, treinar e avaliar
docker compose -f docker-compose.visual.yml run --rm testes             # testes do extra
```

Para refazer a extração: `docker compose -f docker-compose.visual.yml run --rm preparar python -m cardio_visual.data.preparar --sobrescrever`.

### Comandos (sem Docker)

```bash
python -m pip install -e ".[cnn]"
python -m cardio_visual.data.preparar                    # zip do Kaggle -> data/images/<classe>/ e teste cego
python -m cardio_visual.models.predict --demo            # classifica as 4 imagens do teste cego
python -m cardio_visual.models.predict --demo --gabarito # idem, conferindo com data/demo_gabarito.csv
python -m cardio_visual.models.final_model               # avaliação única no teste do modelo final
python -m cardio_visual.models.train --modelo resnet18 --image-size 448 --epochs 20   # retreinar (~25 min na CPU)
python -m cardio_visual.models.cv --modelo resnet18 --image-size 224                  # validação cruzada (~35 min na CPU)
python -m cardio_visual.pipeline                         # caminho rápido (~1 min): teste cego + avaliação, sem treinar
python -m cardio_visual.pipeline --treinar               # caminho completo (~25 min): preparar, treinar e avaliar
pytest tests/test_visual_*.py                            # testes do extra (~2 min; não precisam do dataset)
```

### Resultados

Modelo final: **ResNet18 com transfer learning, 448 px** (`models/cnn_resnet18_448.pt`, já no repositório). Escolhido pela
validação e pela validação cruzada; o teste foi usado uma única vez.

| | Acurácia | F1 macro | Observação |
|---|---|---|---|
| Validação: CNN simples, 224 px | 0,62 | 0,40 | baseline treinado do zero |
| Validação: ResNet18, 224 px | 0,86 | 0,79 | validação cruzada: 0,83 ± 0,05 |
| Validação: ResNet18, 448 px | 0,90 | 0,85 | escolhido |
| **Teste: ResNet18, 448 px** | **0,95** | **0,88** | IC 95% do F1: 0,70 a 0,99; infarto com recall 0,50 (2 de 4) |
| **Teste: normal × alterado** | 0,99 | 0,99 | recall 1,00 e especificidade 0,95 |

Detalhes, interpretação (Grad-CAM) e discussão crítica: seção 10 de `results/relatorio_tecnico.md` e os notebooks
`cnn_01_eda_imagens`, `cnn_02_interpretacao` e `cnn_03_projeto_imagens`.

### Teste rápido (sem treinar)

O modelo final já está no repositório, então não é preciso treinar para testar. Com o zip do Kaggle salvo em
`data/archive.zip` (passo a passo acima), **um comando só** faz tudo em cerca de 1 minuto:

```bash
docker compose -f docker-compose.visual.yml build                # só na primeira vez
docker compose -f docker-compose.visual.yml run --rm rapido      # ou, sem Docker: python -m cardio_visual.pipeline
```

Ele prepara as imagens, classifica as 4 do teste cego (`data/demo/`) conferindo com o gabarito e avalia o modelo final no
teste, sem treinar e sem regravar os resultados versionados. Passos avulsos:

- `python -m cardio_visual.models.predict --demo` classifica as 4 imagens sem mostrar a resposta; com `--gabarito`, confere.
- `python -m cardio_visual.models.predict caminho/da/imagem.jpg` classifica outra imagem.

O resultado é **apoio à decisão**: o modelo não substitui a avaliação do médico, que tem a palavra final.

## Entregáveis (checklist)

- [x] Estrutura do projeto, Dockerfile e README
- [x] Dataset incluído
- [x] EDA (estatísticas, distribuições, correlação)
- [x] Pré-processamento
- [x] 2+ modelos de classificação, com treino / validação / teste
- [x] Avaliação (accuracy, recall, F1) e interpretação (feature importance, SHAP)
- [ ] Relatório técnico (PDF)
- [ ] Vídeo de demonstração (até 15 min)
