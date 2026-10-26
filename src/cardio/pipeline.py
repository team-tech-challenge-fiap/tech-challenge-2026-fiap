"""Pipeline final do projeto (T27): é o comando padrão do Docker.

Executa, em sequência, as duas etapas da avaliação final definidas na T23
(``cardio.models.final_model``):

1. ``train_final_model()``: treina o modelo escolhido na validação (XGBoost, conjunto A) com treino +
   validação e salva ``models/final_model.joblib`` junto com o limiar da T22;
2. ``evaluate_final_model()``: avalia esse modelo **uma única vez** no teste, com o limiar fixo, e grava
   ``results/test_results.csv``.

Uso: ``python -m cardio.pipeline`` (ou ``docker run cardio-classifier``).
"""

import time

from cardio.models.final_model import evaluate_final_model, train_final_model


def main() -> None:
    inicio = time.perf_counter()
    print("=== 1/2 · Treino do modelo final (treino + validação) ===")
    train_final_model()
    print("\n=== 2/2 · Avaliação única no conjunto de teste ===")
    evaluate_final_model()
    print(f"\nPipeline concluído em {time.perf_counter() - inicio:.1f} s.")


if __name__ == "__main__":
    main()
