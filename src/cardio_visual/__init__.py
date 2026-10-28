"""Extra: diagnóstico por imagem de ECG com redes neurais convolucionais (CNN).

Pacote separado do ``cardio`` (projeto tabular), no mesmo nível em ``src/`` e com a mesma organização em camadas.
O extra é opcional e tem dependências próprias (PyTorch), que o projeto principal não precisa instalar.

    config.py                  caminhos e parâmetros do extra (reaproveita SEED e pastas do ``cardio.config``)
    data/
        preparar.py            CNN-02 · extrai o zip do Kaggle numa pasta só por classe e reserva o teste cego
        loader.py              CNN-02 · leitura das pastas e verificação de integridade (corrompidas, duplicadas)
        preprocessing.py       CNN-04 · recorte da grade, split 70/15/15, transformações e DataLoaders
    models/
        estimators/            CNN-05 · um arquivo por modelo: build_model() (cnn_simples, resnet18)
        evaluate.py            CNN-06 · métricas, intervalos de confiança e figuras
        train.py               CNN-07 · treino com early stopping (CLI)
        cv.py                  CNN-07 · validação cruzada estratificada (CLI)
        explain.py             CNN-08 · Grad-CAM e análise de erros
        predict.py             CNN-09 · predição de imagens e teste cego (CLI)
        final_model.py         CNN-09 · avaliação única do modelo final no teste (CLI)
    visualization/
        plots.py               matriz de confusão e curvas ROC, salvas em results/figures/
    pipeline.py                CNN-10 · preparar -> treinar o modelo final -> avaliar no teste (padrão do Docker)

Exige as dependências opcionais: ``pip install -e ".[cnn]"``.
"""
