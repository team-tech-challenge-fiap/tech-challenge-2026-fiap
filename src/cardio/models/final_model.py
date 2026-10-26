"""T23 — Retreinamento e salvamento do modelo final.

Utiliza treino + validação.
Não realiza avaliação no conjunto de teste.
"""

from pathlib import Path

import joblib
import pandas as pd

from cardio.config import MODELS_DIR
from cardio.data.preprocessing import load_splits
from cardio.models.estimators.xgb import build_model
from sklearn.metrics import accuracy_score, roc_auc_score
from cardio.models.evaluate import impact_per_1000

FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"

MODELO_ESCOLHIDO = "xgb"
FEATURE_SET = "A"
LIMIAR_FINAL = 0.36


def train_final_model():
    # Recupera os mesmos splits utilizados pelo projeto.
    X_train, X_val, _, y_train, y_val, _ = load_splits(
        feature_set=FEATURE_SET,
        save_ids=False,
    )

    # Junta treino e validação (85% dos dados).
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    if not X_train_final.index.equals(y_train_final.index):
        raise ValueError("Índices de X e y não estão alinhados.")

    if X_train_final.index.has_duplicates:
        raise ValueError("Existem pacientes duplicados no treino final.")

    # Reconstrói o pipeline selecionado na validação.
    modelo = build_model(X_train_final)

    print(f"Treinando {MODELO_ESCOLHIDO} com conjunto {FEATURE_SET}...")
    print(f"Total de pacientes: {len(X_train_final)}")

    modelo.fit(X_train_final, y_train_final)

    # Salva o modelo e o limiar selecionado na T22.
    artefato = {
        "model": modelo,
        "modelo": MODELO_ESCOLHIDO,
        "features": FEATURE_SET,
        "threshold": LIMIAR_FINAL,
    }

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, FINAL_MODEL_PATH)

    print(f"Modelo final salvo em: {FINAL_MODEL_PATH}")
    print(f"Limiar fixo: {LIMIAR_FINAL}")

    return artefato

def evaluate_final_model():
    """Avalia o modelo final no teste, com limiar fixo da T22."""

    artefato = joblib.load(FINAL_MODEL_PATH)

    modelo = artefato["model"]
    feature_set = artefato["features"]
    limiar = artefato["threshold"]

    # O teste não participa do treinamento nem da escolha do limiar.
    _, _, X_test, _, _, y_test = load_splits(
        feature_set=feature_set,
        save_ids=False,
    )

    probabilidades = modelo.predict_proba(X_test)[:, 1]
    predicoes = (probabilidades >= limiar).astype(int)

    metricas = impact_per_1000(y_test, probabilidades, limiar)

    resultado = {
        "modelo": artefato["modelo"],
        "features": feature_set,
        "n_pacientes": len(y_test),
        "limiar": limiar,
        "accuracy": accuracy_score(y_test, predicoes),
        "auc_roc": roc_auc_score(y_test, probabilidades),
        **metricas,
    }

    caminho = Path("results/test_results.csv")
    caminho.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([resultado]).to_csv(caminho, index=False)

    print("Avaliação final concluída.")
    print(pd.DataFrame([resultado]).to_string(index=False))
    print(f"Resultados salvos em: {caminho}")

    return resultado


if __name__ == "__main__":
    train_final_model()