FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

WORKDIR /app

# Dependências de sistema mínimas (libgomp para scikit-learn/shap)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências primeiro para aproveitar o cache de camadas
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código do projeto
COPY . .

EXPOSE 8888 8000 8501

# Padrão: treina e avalia (na validação) o baseline, a Regressão Logística no conjunto A, e grava
# models/logistic_A.joblib e uma linha em results/val_results.csv.
# Para outro modelo: docker run cardio-classifier python -m cardio.models.train --modelo random_forest --features B
# TODO (T23/T29): trocar pelo pipeline final (todos os modelos + avaliação no teste) quando existir.
CMD ["python", "-m", "cardio.models.train", "--modelo", "logistic", "--features", "A"]
