"""Um arquivo por modelo (T15–T19), seguindo o contrato da T13 (build_model() + PARAM_GRID).

Arquivos:
    example.py         T13 · baseline (DummyClassifier), usado para testar o contrato
    logistic.py        T15 · Regressão Logística (baseline)
    knn.py             T16 · KNN
    random_forest.py   T17 · Random Forest
    xgb.py             T18 · XGBoost  (não usar "xgboost.py": conflita com a biblioteca)
    svm.py             T19 · SVM

Contrato: ``build_model(X_train)`` devolve
``Pipeline([("prep", build_preprocessor(X_train, scale=...)), ("model", Estimador())])`` e
``PARAM_GRID`` usa o prefixo ``model__``. Use ``scale=False`` para modelos de árvore.

O nome do arquivo é o valor de ``--modelo`` em ``python -m cardio.models.train``.
"""
