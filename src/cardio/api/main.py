"""API REST para demonstração do modelo. Docs em http://localhost:8000/docs"""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Cardio IA - Tech Challenge FIAP")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    """TODO equipe: receber os dados do paciente e usar cardio.models.predict."""
    raise HTTPException(status_code=501, detail="Endpoint ainda não implementado")
