# Cardio IA — Tech Challenge FIAP Fase 1

Projeto de análise e classificação de risco cardiovascular usando `cardio_train.csv`
([Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)).

O dataset deve ser um CSV separado por `;` com as colunas `id`, `age`, `gender`, `height`, `weight`,
`ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active` e `cardio`.

## Dataset

O dataset é o **Cardiovascular Disease Dataset** do Kaggle:
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>.
São 70.000 pacientes e 13 colunas (arquivo `data/raw/cardio_train.csv`, separador `;`).

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

## Estrutura

```text
data/
  raw/cardio_train.csv           # dataset original
  processed/                     # dados preparados (gerados)
notebooks/
  00_projeto_final.ipynb         # notebook de ponta a ponta (T29, em construção)
  01_eda_visao_geral.ipynb       # T03: visão geral, alvo e qualidade dos dados
  02_eda_demografia_imc.ipynb    # T04: idade, sexo, altura, peso e IMC
  03_eda_pressao_arterial.ipynb  # T05: pressão arterial
  04_eda_exames_habitos.ipynb    # T06: colesterol, glicose e hábitos de vida
  05_feature_engineering.ipynb   # T10: variáveis derivadas e conjuntos A e B
  06_modelo_regressao_logistica.ipynb  # T15: regressão logística (baseline) e odds ratio
  07_ajuste_limiar.ipynb         # T22: ajuste do limiar de decisão (recall >= 0,80) e impacto por 1.000 pacientes
src/cardio/
  config.py                      # SEED, caminhos e contrato do dataset
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
    explain.py                   # TODO: feature importance / SHAP
  visualization/
    eda.py                       # gráficos da EDA
    plots.py                     # TODO: gráficos para o relatório
  api/
    main.py                      # API FastAPI (/health; /predict TODO)
app/
  streamlit_app.py               # TODO: interface de demonstração (vídeo)
models/                          # modelos gerados (.joblib)
results/
  figures/                       # gráficos gerados (.png)
  relatorio_tecnico.md           # esqueleto do relatório
tests/                           # testes automatizados
Dockerfile · docker-compose.yml · requirements.txt · pyproject.toml
```

## Onde fica cada coisa citada nas issues

As issues foram escritas com uma estrutura simplificada. Os caminhos reais são:

| Na issue | No projeto |
|---|---|
| `data/cardio_train.csv` | `data/raw/cardio_train.csv` |
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

**Treinar e avaliar** (padrão: Regressão Logística, conjunto A, avaliada na validação). Para os resultados
aparecerem na sua máquina, monte as pastas `results/` e `models/`:

```powershell
# Windows (PowerShell)
docker run --rm -v "${PWD}/results:/app/results" -v "${PWD}/models:/app/models" cardio-classifier
```

```bash
# Linux / macOS
docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/models:/app/models" cardio-classifier
```

Saída: `models/logistic_A.joblib` e uma linha nova em `results/val_results.csv`. Para outro modelo ou conjunto,
acrescente o comando no fim:

```bash
docker run --rm -v "${PWD}/results:/app/results" -v "${PWD}/models:/app/models" cardio-classifier \
  python -m cardio.models.train --modelo random_forest --features B
```

**Abrir o Jupyter** e acessar http://localhost:8888 (a pasta do projeto é montada, então o que for salvo nos
notebooks fica na sua máquina):

```powershell
docker run --rm -p 8888:8888 -v "${PWD}:/app" cardio-classifier jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=''
```

**Rodar os testes:**

```bash
docker run --rm cardio-classifier pytest -q
docker run --rm cardio-classifier pytest tests/test_data.py -q   # só qualidade dos dados e vazamento (T12)
```

### Opção 2: `docker compose` (dia a dia do time)

Monta a pasta inteira do projeto como volume, então alterações no código valem sem reconstruir a imagem.

```bash
docker compose build                  # constrói a imagem (repetir só quando o requirements.txt mudar)
docker compose up jupyter             # Jupyter em http://localhost:8888 (Ctrl + C para parar)
docker compose run --rm train         # treina o Random Forest (conjunto A) -> models/random_forest_A.joblib
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

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m cardio.models.train --modelo random_forest --features A   # ou: cardio-train --modelo ...
pytest
pytest tests/test_data.py   # só qualidade dos dados e vazamento (T12)
```

`tests/test_data.py` (T12) garante que `id` e `cardio` nunca entram nas features, que a limpeza não deixa
valores fora das faixas, que treino, validação e teste não têm pacientes em comum, que o split é
reprodutível e estratificado e que o pré-processador é ajustado só no treino.

O treino usa `load_splits()` (70/15/15), avalia na **validação**, salva `models/<modelo>_<A|B>.joblib` e
acrescenta uma linha em `results/val_results.csv`. `--modelo` é o nome de um arquivo em
`src/cardio/models/estimators/`; use `--sample 5000` para testes rápidos.

## Entregáveis (checklist)

- [x] Estrutura do projeto, Dockerfile e README
- [x] Dataset incluído
- [ ] EDA (estatísticas, distribuições, correlação)
- [ ] Pré-processamento
- [ ] 2+ modelos de classificação, com treino / validação / teste
- [ ] Avaliação (accuracy, recall, F1) e interpretação (feature importance, SHAP)
- [ ] Relatório técnico (PDF)
- [ ] Vídeo de demonstração (até 15 min)
