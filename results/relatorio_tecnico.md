# Relatório Técnico — Tech Challenge Fase 1

**Equipe:**
**Repositório:**
**Vídeo:**

## 1. Problema
## 2. Dataset
## 3. Exploração de dados
## 4. Estratégias de pré-processamento
## 5. Modelos usados e por quê



### Random Forest (T17) — Análise da classificação de doenças cardiovasculares

#### Objetivo e justificativa do modelo

O algoritmo Random Forest foi utilizado para desenvolver um modelo de inteligência artificial capaz de identificar padrões associados à presença de doenças cardiovasculares, considerando informações clínicas, características demográficas e hábitos de vida dos pacientes.

O Random Forest é um método de aprendizado de máquina baseado na combinação de múltiplas árvores de decisão. Sua principal vantagem está na capacidade de identificar relações não lineares entre diferentes variáveis, permitindo analisar conjuntamente fatores como idade, pressão arterial, colesterol, glicemia, peso corporal e hábitos de vida.

A escolha desse algoritmo também considerou sua capacidade de reduzir a variabilidade das previsões em comparação com uma árvore de decisão isolada.

#### Desenvolvimento do modelo e controle de sobreajuste

Na etapa inicial do desenvolvimento, foi realizado um experimento utilizando uma amostra de 5.000 pacientes. O modelo apresentou acurácia de 100,00% nos dados de treinamento e 70,88% nos dados de validação.

Essa diferença indicou a ocorrência de sobreajuste (*overfitting*), situação em que o algoritmo apresenta excelente desempenho nos dados utilizados para seu treinamento, mas menor capacidade de generalização para novos registros.

Para reduzir esse comportamento, foram ajustados parâmetros relacionados à complexidade das árvores de decisão, incluindo a profundidade máxima e a quantidade mínima de amostras por folha.

Posteriormente, foram realizados treinamentos utilizando 47.955 registros de pacientes, com avaliação em um conjunto independente de validação contendo 10.277 registros.

Após a otimização completa dos hiperparâmetros, o Modelo A apresentou acurácia de 74,58% no treinamento e 72,86% na validação, representando uma diferença de 1,72 ponto percentual.

O Modelo B apresentou acurácia de 74,54% no treinamento e 72,74% na validação, com diferença de 1,80 ponto percentual.

As diferenças relativamente pequenas entre treinamento e validação sugerem que o controle da complexidade das árvores contribuiu para reduzir o sobreajuste observado no experimento inicial.

#### Metodologia de otimização dos hiperparâmetros (T14)

A otimização dos modelos foi realizada por meio do algoritmo `RandomizedSearchCV`, utilizando validação cruzada estratificada com 5 folds.

Foram avaliadas as 24 combinações possíveis de hiperparâmetros definidas no `PARAM_GRID` do Random Forest, considerando:

- `max_depth`: 6, 10, 14 e sem limite (`None`).
- `min_samples_leaf`: 1, 20 e 50.
- `max_features`: `sqrt` e `0.5`.

A busca foi realizada exclusivamente sobre os 47.955 registros de treinamento, preservando os conjuntos de validação e teste.

A seleção da melhor combinação utilizou como critério principal a área sob a curva ROC (`roc_auc`).

O pré-processamento foi integrado ao pipeline do modelo, permitindo que os ajustes necessários fossem realizados separadamente em cada fold da validação cruzada, reduzindo o risco de vazamento de informações entre os subconjuntos.

Para o Modelo A, a melhor configuração apresentou AUC ROC média de 80,05% na validação cruzada, com desvio-padrão de 0,36 ponto percentual.

Para o Modelo B, a melhor configuração apresentou AUC ROC média de 80,00%, com desvio-padrão de 0,37 ponto percentual.

Após a seleção dos melhores hiperparâmetros, cada modelo foi novamente ajustado utilizando todos os 47.955 registros de treinamento.

O tempo aproximado da busca foi de 214,55 segundos para o Modelo A e 295,63 segundos para o Modelo B.

#### Avaliação e comparação dos conjuntos de atributos

Foram consideradas duas configurações de variáveis:

- **Modelo A:** composto por 11 atributos clínicos, demográficos e comportamentais.
- **Modelo B:** composto por 16 atributos, incluindo variáveis derivadas, como índice de massa corporal (IMC), pressão de pulso e pressão arterial média.

Os dois modelos otimizados foram avaliados sobre os mesmos 10.277 registros de validação, não utilizados no treinamento.

O **Modelo A** apresentou acurácia global de **72,86%**, precisão de **75,23%**, sensibilidade de **67,34%**, F1-score de **71,07%** e AUC ROC de **79,08%**.

O **Modelo B** apresentou acurácia global de **72,74%**, precisão de **74,46%**, sensibilidade de **68,38%**, F1-score de **71,29%** e AUC ROC de **78,96%**.

Os resultados demonstraram desempenho semelhante entre os dois conjuntos de atributos.

O Modelo A apresentou valores ligeiramente superiores de acurácia, precisão e AUC ROC. A diferença de AUC ROC foi de aproximadamente 0,12 ponto percentual, indicando capacidade discriminativa praticamente equivalente entre os modelos na validação realizada.

Por outro lado, o Modelo B apresentou maior sensibilidade e F1-score, sugerindo uma pequena vantagem na identificação de pacientes classificados como positivos na base de referência.

Embora essas diferenças sejam relevantes para a interpretação dos resultados, não é possível afirmar superioridade estatisticamente significativa de um modelo sobre o outro sem análises complementares.

#### Interpretação clínica dos resultados

A avaliação clínica dos modelos deve considerar não apenas a acurácia global, mas também a capacidade de identificar corretamente pacientes com doença cardiovascular registrada na base de dados.

O **Modelo A**, com sensibilidade de **67,34%**, identificou corretamente aproximadamente 67 de cada 100 pacientes positivos.

O **Modelo B**, com sensibilidade de **68,38%**, identificou aproximadamente 68 de cada 100 pacientes positivos.

Portanto, o Modelo B apresentou uma sensibilidade aproximadamente 1,04 ponto percentual superior à do Modelo A.

Na avaliação dos 10.277 registros de validação, o Modelo A apresentou:

- **4.063 verdadeiros negativos:** pacientes sem a condição corretamente classificados.
- **3.425 verdadeiros positivos:** pacientes com a condição corretamente identificados.
- **1.128 falsos positivos:** pacientes sem a condição classificados como positivos.
- **1.661 falsos negativos:** pacientes com a condição classificados como negativos.

O Modelo B apresentou:

- **3.998 verdadeiros negativos:** pacientes sem a condição corretamente classificados.
- **3.478 verdadeiros positivos:** pacientes com a condição corretamente identificados.
- **1.193 falsos positivos:** pacientes sem a condição classificados como positivos.
- **1.608 falsos negativos:** pacientes com a condição classificados como negativos.

Comparativamente, o Modelo B identificou corretamente 53 pacientes positivos a mais do que o Modelo A, reduzindo o número de falsos negativos de 1.661 para 1.608.

Entretanto, essa melhoria foi acompanhada pelo aumento de 65 falsos positivos, indicando uma troca entre sensibilidade e precisão.

Do ponto de vista clínico, a redução dos falsos negativos pode ser particularmente importante em aplicações de triagem, nas quais deixar de identificar pacientes com determinada condição pode representar um risco relevante.

Apesar disso, ambos os modelos deixaram de identificar aproximadamente um terço dos pacientes positivos, demonstrando limitações importantes para aplicações assistenciais.

A AUC ROC de 79,08% no Modelo A e de 78,96% no Modelo B indica capacidade discriminativa moderada a razoável, considerando diferentes limiares de classificação.

Esses resultados demonstram potencial para análise exploratória de dados cardiovasculares, mas não justificam a utilização isolada dos modelos para diagnóstico ou tomada de decisão clínica.

#### Configuração final dos modelos

Após a otimização de hiperparâmetros, ambos os modelos utilizaram 300 árvores de decisão (`n_estimators=300`) e semente aleatória fixa de valor 42 (`random_state=42`), favorecendo a reprodutibilidade dos experimentos.

No **Modelo A**, a busca selecionou:

- `max_depth=10`
- `min_samples_leaf=20`
- `max_features="sqrt"`
- `n_estimators=300`
- `random_state=42`

No **Modelo B**, a busca selecionou:

- `max_depth=10`
- `min_samples_leaf=50`
- `max_features=0.5`
- `n_estimators=300`
- `random_state=42`

A configuração do Modelo A coincidiu com os hiperparâmetros anteriormente utilizados nos experimentos de controle de sobreajuste.

Já o Modelo B apresentou uma configuração distinta, com maior quantidade mínima de amostras por folha, favorecendo árvores menos sensíveis a pequenas variações nos dados.

As configurações descritas foram selecionadas a partir da validação cruzada realizada exclusivamente sobre os dados de treinamento e posteriormente avaliadas no conjunto de validação.

#### Conclusão

O algoritmo Random Forest demonstrou capacidade de identificar padrões associados à presença de doenças cardiovasculares a partir de informações clínicas, demográficas e comportamentais dos pacientes.

A otimização completa dos hiperparâmetros, realizada por meio da T14, permitiu avaliar sistematicamente 24 configurações para cada conjunto de atributos, utilizando validação cruzada estratificada de 5 folds.

Após a otimização, o **Modelo A apresentou a maior AUC ROC (79,08%)**, além de acurácia de 72,86% e precisão de 75,23%.

O **Modelo B apresentou maior sensibilidade (68,38%) e F1-score (71,29%)**, identificando corretamente 53 pacientes positivos adicionais em comparação ao Modelo A.

Os resultados indicam que a inclusão de variáveis derivadas no conjunto B não proporcionou melhoria relevante na capacidade discriminativa global medida pela AUC ROC, mas esteve associada a uma pequena melhora na sensibilidade e no F1-score após a otimização.

Considerando a AUC ROC como critério principal de seleção dos hiperparâmetros, o Modelo A apresentou o melhor resultado entre as configurações avaliadas.

Entretanto, sob uma perspectiva de triagem cardiovascular, o Modelo B merece consideração devido à menor quantidade de falsos negativos, embora apresente maior número de falsos positivos.

As diferenças observadas são pequenas e não permitem estabelecer superioridade estatisticamente significativa sem avaliações adicionais.

Apesar dos resultados, a proporção de falsos negativos evidencia limitações importantes para possíveis aplicações assistenciais.

Portanto, o Random Forest deve ser considerado, nesta etapa, um modelo experimental de classificação e análise de dados, **não constituindo ferramenta diagnóstica clinicamente validada**.

Antes de qualquer aplicação em ambiente clínico, seriam necessárias avaliações adicionais, incluindo validação externa, calibração das probabilidades, investigação de possíveis vieses, análise dos limiares de classificação e estudos sobre as consequências clínicas dos erros de classificação.

Os resultados apresentados correspondem à validação interna dos modelos após a otimização completa de hiperparâmetros.

O conjunto de teste foi preservado e deverá ser utilizado exclusivamente na etapa de avaliação final do projeto (T23).

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

### KNN (T16)

O KNN (*k-nearest neighbors*) classifica cada paciente pela maioria entre os `k` pacientes mais parecidos
do treino, o que é fácil de explicar a um médico ("pacientes parecidos tiveram o mesmo desfecho"). Foi
escolhido como modelo baseado em distância, diferente das árvores e da regressão logística. Como depende
de distâncias, as variáveis contínuas são escalonadas; e, como o dataset é ruidoso, foram testados `k`
bem maiores que o padrão.

**Hiperparâmetros.** Grid completo (`n_neighbors` em 15, 31, 51 e 101; `weights` uniforme ou por distância)
com validação cruzada de 5 folds numa amostra de 10.000 pacientes do treino, escolhendo pela AUC. Os
conjuntos A e B escolheram `n_neighbors = 101` e `weights = "uniform"` (AUC de 0,782 e 0,779 na CV). O
melhor valor está na borda do grid, mas a diferença para `k = 51` é de cerca de 0,003 de AUC.

**Resultados na validação** (o teste não foi usado).

| Conjunto | Acurácia | Precision | Recall | F1-score | AUC ROC |
|---|---|---|---|---|---|
| A (variáveis originais) | 72,49% | 75,34% | 66,02% | 70,38% | 78,53% |
| B (com variáveis derivadas) | 72,40% | 75,33% | 65,79% | 70,24% | 78,62% |

O KNN ficou ligeiramente abaixo do XGBoost (AUC 0,792) e próximo da Regressão Logística (0,783 a 0,785);
as variáveis derivadas do conjunto B não ajudaram. O "treino" leva cerca de 0,1 s (só guarda os dados),
mas a predição na validação (10.277 pacientes) leva cerca de 1,2 a 1,4 s, porque cada paciente é comparado
com todos os do treino: é o modelo mais lento para prever.

## 6. Resultados
## 7. Interpretação (feature importance / SHAP)
## 8. Discussão crítica (uso na prática — o médico tem a palavra final)
## 9. Conclusão
