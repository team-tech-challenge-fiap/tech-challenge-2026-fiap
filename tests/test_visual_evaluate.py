"""Testes da avaliação dos modelos de imagem (CNN-06), com resultados calculáveis à mão."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from cardio_visual.models.evaluate import (  # noqa: E402
    append_result,
    bootstrap_ci,
    evaluate_model,
    evaluate_predictions,
    predict_loader,
    summary_table,
)

CLASSES = ["normal", "infarto", "arritmia"]


def _probs(rotulos, n=3):
    """Probabilidade 1.0 na classe indicada (previsão certa quando igual ao real)."""
    p = np.zeros((len(rotulos), n))
    p[np.arange(len(rotulos)), rotulos] = 1.0
    return p


def test_previsao_perfeita_da_tudo_1() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    m = evaluate_predictions(y, _probs(y), CLASSES)

    assert m["acuracia"] == m["recall_macro"] == m["precision_macro"] == m["f1_macro"] == m["auc_ovr"] == 1.0
    assert m["por_classe"]["suporte"].tolist() == [2, 2, 2]


def test_metricas_por_classe_calculadas_a_mao() -> None:
    # classe 0: 4 reais, 3 acertos (1 previsto como 1); classe 1: 2 reais, 2 acertos e 1 falso positivo vindo da 0
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate_predictions(y, _probs(pred, 2), ["a", "b"])
    c = m["por_classe"].set_index("classe")

    assert c.loc["a", "recall"] == pytest.approx(3 / 4) and c.loc["a", "precision"] == pytest.approx(1.0)
    assert c.loc["b", "recall"] == pytest.approx(1.0) and c.loc["b", "precision"] == pytest.approx(2 / 3)
    assert m["recall_macro"] == pytest.approx((3 / 4 + 1.0) / 2)
    assert m["acuracia"] == pytest.approx(5 / 6)
    assert m["f1_macro"] == pytest.approx((2 * 1.0 * 0.75 / 1.75 + 2 * (2 / 3) * 1.0 / (5 / 3)) / 2)


def test_classe_ausente_nao_conta_na_media_e_auc_vira_nan() -> None:
    y = np.array([0, 0, 1, 1])  # a classe 2 não aparece
    m = evaluate_predictions(y, _probs(y), CLASSES)

    assert m["recall_macro"] == 1.0  # a média ignora a classe sem exemplos
    assert np.isnan(m["auc_ovr"])
    assert m["por_classe"].set_index("classe").loc["arritmia", "suporte"] == 0


def test_entrada_com_numero_de_colunas_errado_levanta_erro() -> None:
    with pytest.raises(ValueError, match="colunas"):
        evaluate_predictions(np.array([0, 1]), np.ones((2, 5)) / 5, CLASSES)


def test_bootstrap_ci_contem_a_estimativa_e_e_reprodutivel() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 120)
    pred = np.where(rng.random(120) < 0.75, y, rng.integers(0, 3, 120))
    prob = _probs(pred)
    ponto = evaluate_predictions(y, prob, CLASSES)["recall_macro"]

    baixo, alto = bootstrap_ci(y, prob, "recall_macro", n_boot=300)

    assert baixo <= ponto <= alto and alto - baixo > 0.02  # com 120 imagens o intervalo não é estreito
    assert bootstrap_ci(y, prob, "recall_macro", n_boot=300) == (baixo, alto)
    assert bootstrap_ci(y, prob, "f1_macro", n_boot=50)[0] <= 1.0
    with pytest.raises(ValueError):
        bootstrap_ci(y, prob, "auc", n_boot=10)


def test_intervalo_e_mais_estreito_com_mais_imagens() -> None:
    rng = np.random.default_rng(1)

    def largura(n):
        y = rng.integers(0, 2, n)
        pred = np.where(rng.random(n) < 0.8, y, 1 - y)
        b, a = bootstrap_ci(y, _probs(pred, 2), "recall_macro", n_boot=300)
        return a - b

    assert largura(2000) < largura(100)


class _Constante(torch.nn.Module):
    """Modelo de teste: sempre prevê a classe 1 com logits fixos."""

    def forward(self, x):
        return torch.tensor([[0.0, 2.0, 0.0]]).repeat(len(x), 1)


@pytest.fixture()
def loader():
    x = torch.randn(12, 3, 8, 8)
    y = torch.tensor([0, 1, 2] * 4)
    return DataLoader(TensorDataset(x, y), batch_size=5)


def test_predict_loader_devolve_probabilidades_validas(loader) -> None:
    y_true, y_prob = predict_loader(_Constante(), loader)

    assert y_true.tolist() == [0, 1, 2] * 4
    assert y_prob.shape == (12, 3) and np.allclose(y_prob.sum(axis=1), 1.0)
    assert (y_prob.argmax(axis=1) == 1).all()


def test_predict_loader_poe_o_modelo_em_modo_de_avaliacao(loader) -> None:
    modelo = _Constante().train()
    predict_loader(modelo, loader)
    assert not modelo.training


def test_evaluate_model_salva_figuras_e_da_intervalos(loader, tmp_path) -> None:
    r = evaluate_model(_Constante(), loader, CLASSES, name="const", split="val", figures_dir=tmp_path)

    assert r["recall_macro"] == pytest.approx(1 / 3)  # só acerta a classe 1
    assert 0 <= r["recall_macro_ic"][0] <= r["recall_macro_ic"][1] <= 1 and len(r["f1_macro_ic"]) == 2
    assert (tmp_path / "cnn_cm_const_val.png").stat().st_size > 0
    assert (tmp_path / "cnn_roc_const_val.png").stat().st_size > 0


def test_evaluate_model_sem_nome_nao_salva_figuras(loader, tmp_path) -> None:
    evaluate_model(_Constante(), loader, CLASSES, figures_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_append_result_e_summary_table(tmp_path) -> None:
    csv = tmp_path / "res.csv"
    append_result({"modelo": "cnn_simples", "f1_macro": 0.50, "recall_macro": 0.5}, csv)
    append_result({"modelo": "resnet18", "f1_macro": 0.80, "recall_macro": 0.8}, csv)
    append_result({"modelo": "cnn_simples", "f1_macro": 0.55, "recall_macro": 0.5}, csv)  # treino mais recente

    assert csv.read_text(encoding="utf-8").count("modelo,") == 1  # cabeçalho uma vez só
    tabela = summary_table(csv)
    assert tabela["modelo"].tolist() == ["resnet18", "cnn_simples"]
    assert tabela.loc[tabela["modelo"] == "cnn_simples", "f1_macro"].item() == 0.55
    assert isinstance(tabela, pd.DataFrame)
