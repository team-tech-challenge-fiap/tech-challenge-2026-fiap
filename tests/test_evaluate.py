"""Testes da avaliação padronizada (T20)."""

import joblib
import pandas as pd
import pytest

from cardio.data.preprocessing import load_splits
from cardio.models.estimators import logistic
from cardio.models.evaluate import (
    BASELINE,
    baseline_results,
    evaluate,
    evaluate_saved_model,
    summary_table,
)


@pytest.fixture(scope="module")
def dados():
    X_train, X_val, _, y_train, y_val, _ = load_splits("A", save_ids=False)
    modelo = logistic.build_model(X_train.head(3000)).fit(X_train.head(3000), y_train.head(3000))
    return modelo, X_val, y_val


def test_evaluate_devolve_as_cinco_metricas(dados) -> None:
    modelo, X, y = dados
    metricas = evaluate(modelo, X, y)

    assert set(metricas) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= valor <= 1 for valor in metricas.values())
    assert metricas["accuracy"] > 0.65  # bem acima do baseline (~50%)


def test_f1_e_a_media_harmonica_de_precision_e_recall(dados) -> None:
    modelo, X, y = dados
    m = evaluate(modelo, X, y)
    assert m["f1"] == pytest.approx(2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]))


def test_evaluate_sem_nome_nao_salva_figura(dados, tmp_path) -> None:
    modelo, X, y = dados
    evaluate(modelo, X, y, figures_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_evaluate_com_nome_salva_a_matriz_de_confusao(dados, tmp_path) -> None:
    modelo, X, y = dados
    evaluate(modelo, X, y, name="logistic_A", split="val", figures_dir=tmp_path)
    figura = tmp_path / "cm_logistic_A_val.png"
    assert figura.exists() and figura.stat().st_size > 0


def test_evaluate_saved_model_le_o_conjunto_pelo_nome(dados, tmp_path) -> None:
    modelo, _, _ = dados
    caminho = tmp_path / "logistic_A.joblib"
    joblib.dump(modelo, caminho)

    metricas = evaluate_saved_model(caminho, figures_dir=tmp_path)

    assert metricas["roc_auc"] > 0.7
    assert (tmp_path / "cm_logistic_A_val.png").exists()


@pytest.mark.parametrize("nome", ["modelo.joblib", "modelo_C.joblib"])
def test_evaluate_saved_model_rejeita_nome_sem_conjunto(tmp_path, nome) -> None:
    with pytest.raises(ValueError):
        evaluate_saved_model(tmp_path / nome)


def test_baseline_fica_perto_de_50_por_cento() -> None:
    base = baseline_results()
    assert base["modelo"] == BASELINE
    assert 0.45 < base["acuracia"] < 0.55
    assert 0.45 < base["auc_roc"] < 0.55


def test_summary_table_junta_modelos_dedup_e_inclui_o_baseline(tmp_path) -> None:
    csv = tmp_path / "val_results.csv"
    pd.DataFrame([
        {"modelo": "a", "features": "A", "acuracia": 0.7, "f1_score": 0.7, "recall_score": 0.7,
         "precision_score": 0.7, "auc_roc": 0.70, "tempo_treinamento": 1},
        {"modelo": "a", "features": "A", "acuracia": 0.7, "f1_score": 0.7, "recall_score": 0.7,
         "precision_score": 0.7, "auc_roc": 0.80, "tempo_treinamento": 1},  # treino mais recente
        {"modelo": "b", "features": "B", "acuracia": 0.7, "f1_score": 0.7, "recall_score": 0.7,
         "precision_score": 0.7, "auc_roc": 0.75, "tempo_treinamento": 1},
    ]).to_csv(csv, index=False)

    tabela = summary_table(csv)

    assert len(tabela) == 3
    assert tabela["modelo"].tolist()[:2] == ["a", "b"]  # ordenado pela AUC; "a" ficou com a linha mais recente
    assert tabela.loc[tabela["modelo"] == "a", "auc_roc"].item() == 0.80
    assert BASELINE in tabela["modelo"].tolist()
    assert len(summary_table(csv, include_baseline=False)) == 2


def test_summary_table_com_o_csv_real_tem_baseline_embaixo() -> None:
    tabela = summary_table()
    assert tabela["modelo"].iloc[-1] == BASELINE
