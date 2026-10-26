# Relatório Técnico — Tech Challenge Fase 1

**Equipe:**
**Repositório:**
**Vídeo:**

## 1. Problema
## 2. Dataset
## 3. Exploração de dados
## 4. Estratégias de pré-processamento
## 5. Modelos usados e por quê

### Random Forest (T17)

> Texto preliminar da T32: será atualizado após o tuning (T14) e a comparação entre os conjuntos de features A e B.

O algoritmo Random Forest foi selecionado por sua capacidade de identificar relações não lineares entre
as características dos pacientes e a presença de doença cardiovascular. Por combinar múltiplas árvores de
decisão, pode apresentar maior robustez e capacidade de generalização do que uma árvore de decisão isolada.

**Experimento inicial.** O modelo foi treinado inicialmente com 5.000 pacientes, com sinais de overfitting:
acurácia de 100,00% no treinamento e de 70,88% na validação. Para reduzir o overfitting, foram testados
`n_estimators = 300`, `max_depth = 10` e `min_samples_leaf = 20`.

**Resultados com a base completa (features A, 47.955 pacientes no treinamento).**

| Métrica | Valor |
|---|---|
| Acurácia no treinamento | 74,62% |
| Acurácia na validação | 72,71% |
| Precision | 75,10% |
| Recall | 67,09% |
| F1-score | 70,87% |
| AUC ROC | 79,10% |

A diferença entre as acurácias de treinamento e validação foi de 1,91 ponto percentual, sugerindo uma
redução significativa do overfitting observado inicialmente.

**Pendências.** Comparar com o conjunto de features B (T10) e rodar a busca de hiperparâmetros (T14).
Os resultados apresentados são preliminares.

### XGBoost (T18)

O XGBoost é um conjunto de árvores de decisão construídas em sequência, em que cada árvore nova corrige
os erros das anteriores (*gradient boosting*). Foi escolhido porque costuma ser o modelo mais forte em
dados tabulares desse tamanho e porque funciona bem com o SHAP (T25), que mostra o que pesou na decisão
para cada paciente. Como pode decorar o ruído do dataset, foi usado com árvores rasas e aprendizado lento.

**Hiperparâmetros.** Foram sorteadas 20 combinações do grid com validação cruzada de 5 folds **só no
treino** (47.955 pacientes), escolhendo pela AUC. Os conjuntos A e B escolheram a mesma combinação:
`n_estimators = 200`, `max_depth = 4`, `learning_rate = 0,03`, `subsample = 0,8` e `colsample_bytree = 0,8`
(AUC de 0,801 na validação cruzada). As 8 melhores combinações ficaram entre 0,8005 e 0,8014, então o
resultado é estável.

**Resultados na validação** (o teste não foi usado).

| Conjunto | Acurácia | Precision | Recall | F1-score | AUC ROC |
|---|---|---|---|---|---|
| A (variáveis originais) | 72,63% | 74,53% | 67,89% | 71,06% | 79,19% |
| B (com variáveis derivadas) | 72,51% | 74,43% | 67,72% | 70,92% | 79,16% |

A acurácia no treino foi de 73,9% nos dois conjuntos, de 1,2 a 1,4 ponto acima da validação, sem sinal de
overfitting. O treino leva menos de 1 segundo. As variáveis derivadas do conjunto B (IMC, pressão de
pulso, PAM e as categorias) **não melhoraram** o modelo: a diferença é de 0,03 ponto na AUC. Isso é
esperado, porque as árvores já combinam as variáveis originais. Na validação, o XGBoost ficou praticamente
empatado com o Random Forest (AUC 0,791) e um pouco acima da Regressão Logística (0,783 a 0,785).

## 6. Resultados
## 7. Interpretação (feature importance / SHAP)
## 8. Discussão crítica (uso na prática — o médico tem a palavra final)
## 9. Conclusão
