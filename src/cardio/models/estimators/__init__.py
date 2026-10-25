"""Um arquivo por modelo (T15–T19), seguindo o contrato da T13 (build_model() + PARAM_GRID).

Arquivos:
    example.py         T13 · baseline (DummyClassifier), usado para testar o contrato
    logistic.py        T15 · Regressão Logística (baseline)
    knn.py             T16 · KNN
    random_forest.py   T17 · Random Forest
    xgb.py             T18 · XGBoost  (não usar "xgboost.py": conflita com a biblioteca)
    svm.py             T19 · SVM

O nome do arquivo é o valor de ``--modelo`` em ``python -m cardio.models.train``.
"""
