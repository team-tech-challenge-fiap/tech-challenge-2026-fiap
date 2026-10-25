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
## 6. Resultados
## 7. Interpretação (feature importance / SHAP)
## 8. Discussão crítica (uso na prática — o médico tem a palavra final)
## 9. Conclusão
