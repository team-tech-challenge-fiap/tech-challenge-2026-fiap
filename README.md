# Cardio IA — Tech Challenge FIAP Fase 1

Projeto de análise e classificação de risco cardiovascular usando `cardio_train.csv`
([Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)).

O dataset deve ser um CSV separado por `;` com as colunas `id`, `age`, `gender`, `height`, `weight`,
`ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active` e `cardio`.

## Estrutura

```text
data/
  raw/cardio_train.csv           # dataset original
  processed/                     # dados preparados (gerados)
notebooks/
  01_eda.ipynb                   # análise exploratória
src/cardio/
  config.py                      # caminhos e contrato do dataset
  data/
    loader.py                    # leitura + validação das colunas
    preprocessing.py             # limpeza dos dados
  features/
    build.py                     # engenharia de atributos (idade em anos, IMC)
  models/
    train.py                     # treino (baseline: regressão logística)
    evaluate.py                  # métricas
    predict.py                   # predição com modelo salvo
    explain.py                   # TODO: feature importance / SHAP
  visualization/
    plots.py                     # TODO: gráficos para o relatório
  api/
    main.py                      # API FastAPI (/health; /predict TODO)
app/
  streamlit_app.py               # TODO: interface de demonstração (vídeo)
models/                          # modelos gerados (.joblib)
reports/
  figures/                       # gráficos gerados
  relatorio_tecnico.md           # esqueleto do relatório
tests/                           # testes automatizados
Dockerfile · docker-compose.yml · requirements.txt · pyproject.toml
```

## Executando com Docker

Pré-requisito: [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
docker compose build                  # constrói a imagem
docker compose run --rm train         # treina o modelo -> models/cardio_model.joblib
docker compose run --rm tests         # roda os testes
docker compose up jupyter api app     # sobe os serviços
```

| Serviço | URL |
|---|---|
| Jupyter | http://localhost:8888 |
| API | http://localhost:8000/docs |
| App (demo) | http://localhost:8501 |

A pasta do projeto é montada como volume: o que for gerado no container aparece em `models/` e `reports/figures/`.

## Executando sem Docker

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
cardio-train --dataset data/raw/cardio_train.csv --output-dir models
pytest
```

## Entregáveis (checklist)

- [x] Estrutura do projeto, Dockerfile e README
- [x] Dataset incluído
- [ ] EDA (estatísticas, distribuições, correlação)
- [ ] Pré-processamento
- [ ] 2+ modelos de classificação, com treino / validação / teste
- [ ] Avaliação (accuracy, recall, F1) e interpretação (feature importance, SHAP)
- [ ] Relatório técnico (PDF)
- [ ] Vídeo de demonstração (até 15 min)
