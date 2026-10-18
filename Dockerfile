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

# Padrão: executa o pipeline completo de treino. Os serviços do compose sobrescrevem o comando.
CMD ["python", "-m", "cardio.models.train"]
