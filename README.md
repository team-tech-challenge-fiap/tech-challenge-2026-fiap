# Cardio IA

Projeto de análise e classificação de risco cardiovascular usando `cardio_train.csv`.

## Estrutura

```text
data/raw/          # dataset original
data/processed/    # dados preparados
notebooks/         # exploração
src/cardio/        # código da aplicação
models/            # modelos gerados
reports/figures/   # gráficos gerados
tests/              # testes automatizados
```

## Instalação

```powershell
python -m pip install -e ".[dev]"
```

## Treinamento

```powershell
cardio-train --dataset data/raw/cardio_train.csv --output-dir models
```

O dataset deve ser um CSV separado por `;` com as colunas `id`, `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active` e `cardio`.
