"""Testes do pipeline do extra (CNN-10): os dois caminhos, a ordem das etapas e os argumentos, sem treinar."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from cardio_visual import pipeline  # noqa: E402
from cardio_visual.models.predict import FINAL_CHECKPOINT  # noqa: E402

RESULTADO_FALSO = {
    "por_classe": pd.DataFrame({"classe": ["a"], "recall": [1.0]}),
    "acuracia": 1.0, "f1_macro": 1.0, "f1_macro_ic": (1.0, 1.0), "recall_alterado": 1.0, "especificidade": 1.0,
}


@pytest.fixture()
def chamadas(monkeypatch):
    registro = []
    monkeypatch.setattr(pipeline, "prepare_images", lambda zip_path: registro.append(("preparar", Path(zip_path))))
    monkeypatch.setattr(pipeline.train, "main", lambda argv: registro.append(("treino", argv)))
    monkeypatch.setattr(pipeline.final_model, "main", lambda argv: registro.append(("final", argv)))
    monkeypatch.setattr(pipeline.predict, "main", lambda argv: registro.append(("teste_cego", argv)))

    def avaliacao(checkpoint, **caminhos):
        registro.append(("avaliacao", checkpoint, caminhos))
        return RESULTADO_FALSO

    monkeypatch.setattr(pipeline.final_model, "evaluate_final", avaliacao)
    return registro


def test_padrao_e_o_caminho_rapido_sem_treinar(chamadas) -> None:
    pipeline.main([])
    assert [c[0] for c in chamadas] == ["preparar", "teste_cego", "avaliacao"]  # sem treino


def test_caminho_rapido_usa_o_modelo_do_repositorio_e_confere_o_gabarito(chamadas) -> None:
    pipeline.main([])
    teste_cego = chamadas[1][1]
    assert "--demo" in teste_cego and "--gabarito" in teste_cego
    assert teste_cego[teste_cego.index("--checkpoint") + 1] == str(FINAL_CHECKPOINT)
    assert chamadas[2][1] == FINAL_CHECKPOINT


def test_caminho_rapido_nao_regrava_os_resultados_versionados(chamadas) -> None:
    pipeline.main([])
    caminhos = chamadas[2][2]
    resultados = Path(pipeline.__file__).resolve().parents[2] / "results"
    for caminho in (caminhos["results_path"], caminhos["por_classe_path"], caminhos["figures_dir"]):
        assert resultados not in Path(caminho).resolve().parents and Path(caminho).resolve() != resultados


def test_treinar_executa_preparar_treinar_e_avaliar_nessa_ordem(chamadas) -> None:
    pipeline.main(["--treinar"])
    assert [c[0] for c in chamadas] == ["preparar", "treino", "final"]


def test_treinar_usa_a_configuracao_final_e_avalia_o_checkpoint_dela(chamadas) -> None:
    pipeline.main(["--treinar", "--epochs", "3"])
    treino = dict(zip(chamadas[1][1][::2], chamadas[1][1][1::2], strict=True))

    assert treino == {"--modelo": pipeline.MODELO_FINAL, "--image-size": str(pipeline.IMAGE_SIZE_FINAL),
                      "--epochs": "3"}
    assert chamadas[2][1] == ["--checkpoint", str(FINAL_CHECKPOINT)]
    assert FINAL_CHECKPOINT.name == f"cnn_{pipeline.MODELO_FINAL}_{pipeline.IMAGE_SIZE_FINAL}.pt"  # nomes coerentes


def test_usa_o_zip_informado(chamadas, tmp_path) -> None:
    pipeline.main(["--zip", str(tmp_path / "outro.zip")])
    assert chamadas[0] == ("preparar", tmp_path / "outro.zip")
