# Relatório Técnico — Tech Challenge Fase 1

**Equipe:**
**Repositório:**
**Vídeo:**

## 1. Problema

As doenças cardiovasculares são a principal causa de morte no mundo, segundo a Organização Mundial da
Saúde. Muitos casos só são descobertos quando já há sintomas graves, porque o diagnóstico depende de exames
que nem sempre estão ao alcance de todos os pacientes.

Este projeto constrói um modelo de **triagem**: a partir de dados simples, que qualquer consulta de rotina
coleta (idade, sexo, altura, peso, pressão arterial, colesterol, glicose e hábitos de vida), ele estima se o
paciente tem doença cardiovascular. O objetivo não é diagnosticar, e sim **priorizar** quais pacientes devem
fazer exames mais completos primeiro. Por isso o erro mais grave é o falso negativo (um doente classificado
como saudável), e o **recall** é a métrica mais importante, acompanhado de precision, F1 e AUC. O médico
mantém a decisão final.

## 2. Dataset

Os dados são do *Cardiovascular Disease Dataset*, do Kaggle
(<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>): **70.000 pacientes** e 13
colunas (arquivo `data/raw/cardio_train.csv`, separador `;`). O `id` é só um identificador e `cardio` é o
alvo (1 = doença cardiovascular), o que deixa **11 variáveis** para o modelo, de três tipos:

| Tipo | Variáveis | Observação |
|---|---|---|
| **Objetivas** | idade, sexo, altura, peso | medidas ou registradas pela clínica; a idade vem em dias e é convertida para anos |
| **De exame** | pressão sistólica e diastólica, colesterol, glicose | pressão em mmHg; colesterol e glicose em 3 níveis (normal, acima, muito acima) |
| **Subjetivas** | fumo, álcool, atividade física | **autodeclaradas** (sim/não), logo menos confiáveis |

O alvo é **balanceado**: 35.021 pacientes sem doença e 34.979 com doença (≈ 50% / 50%). Por isso a acurácia
também é informativa e não foi preciso reamostrar. Os pacientes têm de 29,6 a 64,9 anos, e 65% são mulheres.
Não há valores ausentes. O principal problema do dataset é a **qualidade dos registros** (seção 3).

## 3. Exploração de dados

Os detalhes ficam nos notebooks `01_eda_visao_geral.ipynb` (T03), `02_eda_demografia_imc.ipynb` (T04),
`03_eda_pressao_arterial.ipynb` (T05), `04_eda_exames_habitos.ipynb` (T06) e `08_eda_correlacao.ipynb`
(T07); as figuras estão em `results/figures/`.

### 3.1 Problemas de qualidade

- **Duplicados:** 24 linhas idênticas se ignorarmos o `id` (48 registros). Mantê-las colocaria o mesmo
  paciente no treino e na validação, inflando o resultado.
- **Pressão arterial impossível:** a sistólica vai de −150 a 16.020 mmHg e a diastólica de −70 a 11.000, e
  há pacientes com diastólica maior ou igual à sistólica.
- **Altura e peso fora da realidade:** altura de 55 a 250 cm e peso a partir de 10 kg, além de combinações
  que dão um IMC impossível.
- **Tratamento (T08):** `clean()` remove os registros com erro, em vez de corrigi-los, usando as faixas
  plausíveis do `config.py`. O dataset passa de **70.000 para 68.509 linhas** (1.491 removidas, 2,1%), e a
  taxa de doença quase não muda (≈ 50%), então a limpeza não enviesa o alvo.
  Figuras: `t03_alvo.png`, `t03_faixas.png` e `t03_limpeza.png`.

### 3.2 O que se relaciona com a doença

Os números abaixo são sobre os 68.509 pacientes limpos; a taxa de doença é a proporção com `cardio = 1`.
As correlações são de Pearson, como na T07. Figuras: `t07_ranking_alvo.png` (correlação de cada variável com a
doença), `t07_taxa_por_faixa.png` (taxa de doença por faixa de idade e de pressão, que mostra a forma da relação)
e `t06_colesterol_glicose.png` (colesterol e glicose).

| Fator | Achado |
|---|---|
| **Pressão arterial** (o fator mais forte) | Sistólica e diastólica são as variáveis mais correlacionadas com o alvo (correlação de Pearson de 0,43 e 0,34). Pela classificação ACC/AHA, a taxa de doença é de 22% com pressão normal, 32% na elevada, 39% na hipertensão 1 e **78% na hipertensão 2**. |
| **Idade** | A doença cresce de forma contínua com a idade: 24% até 40 anos, 43% entre 45 e 50, 56% entre 55 e 60 e **67% depois dos 60** (correlação de 0,24). |
| **Colesterol** | 44% de doença com colesterol normal, 60% acima do normal e **76% muito acima** (correlação de 0,22). A glicose segue o mesmo sentido, mais fraca: 48%, 59% e 62% (0,09). |
| **Peso e IMC** | A taxa é de 27% abaixo do peso, 40% no peso normal, 51% no sobrepeso e **63% na obesidade** (correlação do IMC de 0,19). A altura isoladamente não tem relação (−0,01). |
| **Sexo** | Praticamente nenhuma: 49% das mulheres e 50% dos homens têm doença (correlação de 0,01). Vale a pena lembrar que 65% da amostra são mulheres. |
| **Hábitos** | Efeito fraco e, no fumo e no álcool, **contrário ao esperado**: 47% de doença entre fumantes contra 50% entre não fumantes. Quem faz atividade física tem 49% contra 53% de quem não faz. |

### 3.3 Cuidados com a interpretação

- **Hábitos autodeclarados não são confiáveis.** O resultado "fumar protege o coração" não indica que fumar
  faça bem: o sinal vem de como a variável foi coletada (subnotificação, quem parou de fumar depois do
  diagnóstico e uma medida só de sim/não). A T06 testou o efeito de sexo e idade e ele explica só uma parte (figuras `t06_habitos.png` e
  `t06_fumo_estratificado.png`).
  Essas variáveis ficam no modelo pelo valor preditivo, mas o efeito delas **não deve ser lido como causal**.
- **Nenhuma variável isolada separa os grupos:** mesmo a pressão, a mais forte, deixa 22% de doentes entre
  quem tem pressão normal. Isso explica o teto de desempenho dos modelos (acurácia em torno de 73%, AUC em
  torno de 0,79) e é discutido na análise de erros (T26).
- **Não há vazamento de dados (T07):** o `id` não tem relação com a doença (correlação de 0,004 e AUC de 0,50
  usando só o `id`) e fica fora das variáveis do modelo, e nenhuma variável é consequência direta do
  diagnóstico. A falta de informação sobre o tratamento do paciente é uma limitação do dataset, e não vazamento.
## 4. Estratégias de pré-processamento

O pré-processamento transforma o arquivo bruto (70.000 pacientes) em dados prontos para os modelos. Ele tem
quatro etapas: limpeza dos registros impossíveis, conversão e criação de variáveis, divisão em treino,
validação e teste, e preparação das variáveis dentro do pipeline de cada modelo. Todas usam as mesmas
funções do código (`clean()`, `load_splits()` e `build_preprocessor()`), então os cinco integrantes treinaram
os modelos sobre exatamente os mesmos dados.

### 4.1 Limpeza: registros impossíveis e duplicados (T08)

O dataset não tem valores ausentes, mas tem **valores fisiologicamente impossíveis**, típicos de erro de
digitação: pressão sistólica de −150 ou 16.020 mmHg, diastólica de 11.000 mmHg, adultos com 55 cm de altura
ou 10 kg de peso. Esses registros foram **removidos**, e não corrigidos: na maioria dos casos não há como
saber qual era o valor certo (um "1100" pode ser 110 ou 100), e corrigir por suposição colocaria no modelo
dados inventados. As faixas aceitas foram definidas nas análises exploratórias (T04 e T05):

| Regra (aplicada em sequência) | Faixa aceita | Justificativa clínica | Linhas removidas |
|---|---|---|---|
| Duplicados | — | Mesmo paciente registrado duas vezes (todas as colunas iguais, exceto o `id`) | 24 |
| Pressão sistólica | 70 a 250 mmHg | Abaixo de 70 indica choque circulatório; acima de 250 é extremamente raro e, aqui, eram erros de digitação | 229 |
| Pressão diastólica | 40 a 200 mmHg | Fora disso são valores digitados com zeros a mais ou negativos | 997 |
| Diastólica < sistólica | — | A pressão "de baixo" nunca é maior ou igual à "de cima" | 103 |
| Altura | 130 a 210 cm | Adultos fora dessa faixa são raríssimos; os casos encontrados eram erros | 89 |
| Peso | 35 a 200 kg | Abaixo de 35 kg é incompatível com adultos dessas alturas | 16 |
| IMC | 15 a 60 kg/m² | Pega combinações de altura e peso impossíveis que passam nas regras anteriores | 33 |
| **Total** | | | **1.491 (2,1%)** |

As regras são aplicadas uma após a outra, e cada linha da tabela conta só os registros que aquela regra
removeu, por isso a soma bate com o total. Os **24 duplicados** (registros com todas as informações iguais,
mudando só o número de identificação) foram removidos primeiro, para que o mesmo paciente não apareça ao
mesmo tempo no treino e no teste, o que inflaria artificialmente o resultado.

Depois da limpeza ficam **68.509 pacientes**. A proporção de doentes praticamente não muda (de 49,97% para
49,49%), o que mostra que os registros removidos estavam espalhados entre os dois grupos e a limpeza não
desequilibrou as classes. As faixas ficam centralizadas no código (`config.py`), e testes automatizados
confirmam que nenhum valor fora delas sobra depois da limpeza.

### 4.2 Conversões e variáveis derivadas (T02 e T10)

Duas colunas mudam de formato: a **idade**, que vem em dias no arquivo original, é convertida para anos, e o
**sexo** (1 = mulher, 2 = homem) vira uma variável `sex_male` (1 = homem, 0 = mulher), mais fácil de
interpretar. O identificador do paciente nunca é usado como variável.

Além das variáveis originais, foram criadas cinco **variáveis derivadas** com significado clínico. Elas são
calculadas só a partir das próprias medidas do paciente, nunca a partir do diagnóstico, por isso não "entregam
a resposta" ao modelo:

| Variável | Como é calculada | Por que pode ajudar |
|---|---|---|
| IMC | peso ÷ altura² | Resume peso e altura num único indicador de excesso de peso, fator de risco conhecido |
| Pressão de pulso | sistólica − diastólica | Diferença alta indica rigidez das artérias, comum no envelhecimento vascular |
| Pressão arterial média (PAM) | (sistólica + 2 × diastólica) ÷ 3 | Representa a pressão com que o sangue chega aos órgãos |
| Categoria de pressão | Normal, elevada, hipertensão 1 ou 2 (diretriz ACC/AHA 2017) | Traduz os números na classificação usada na prática médica |
| Faixa de IMC | Abaixo do peso, normal, sobrepeso ou obesidade (OMS) | Idem, para o peso |

Com isso, cada modelo foi treinado com dois **conjuntos de variáveis**: o **A**, com as 11 variáveis originais
já limpas (idade, sexo, altura, peso, pressões, colesterol, glicose, fumo, álcool e atividade física), e o **B**,
com o A mais as 5 derivadas. Comparar os dois mostra se a engenharia de variáveis traz ganho real. Nos
resultados (seção 5), a diferença entre A e B ficou abaixo de 0,01 de AUC em todos os modelos: as derivadas
são combinações das variáveis originais, que os modelos já conseguem aproveitar sozinhos.

### 4.3 Divisão em treino, validação e teste (T09)

Os 68.509 pacientes foram divididos uma única vez, com sorteio fixo (semente 42), em três grupos:

| Conjunto | Pacientes | Proporção | Uso |
|---|---|---|---|
| Treino | 47.955 | 70% | Os modelos aprendem com esses pacientes |
| Validação | 10.277 | 15% | Comparar modelos, escolher hiperparâmetros e o limiar de decisão |
| Teste | 10.277 | 15% | Avaliação final, **usado uma única vez** |

A divisão é **estratificada**: os três grupos têm a mesma proporção de doentes (49,5%), então nenhum fica
mais "fácil" que outro. Ela é sempre a mesma, para todos os integrantes e em qualquer computador, e o grupo de
cada paciente fica registrado em `results/split_ids.csv`.

O **teste só é usado no final** (T23) porque é o único grupo que simula pacientes nunca vistos. Se fosse
usado para escolher o modelo ou ajustar parâmetros, o resultado final ficaria otimista demais, como um aluno
que estuda com as questões da própria prova. Por isso, todas as escolhas (hiperparâmetros, conjunto de
variáveis e limiar) foram feitas com o treino e a validação. A busca de hiperparâmetros usou, além disso,
**validação cruzada de 5 partes dentro do treino**: o treino é dividido em 5 pedaços, e cada combinação é
testada 5 vezes, cada vez deixando um pedaço de fora, o que dá uma estimativa mais estável.

### 4.4 Preparação das variáveis dentro do pipeline (T11)

Antes de chegar ao algoritmo, cada tipo de variável recebe um tratamento:

- **Contínuas** (idade, altura, peso, pressões, IMC, pressão de pulso e PAM): são **padronizadas**, isto é,
  colocadas na mesma escala (média 0 e desvio-padrão 1). Sem isso, modelos baseados em distância ou em
  coeficientes, como KNN e Regressão Logística, dariam mais peso ao peso em kg ou à pressão em mmHg só porque
  os números são maiores. Os modelos de árvore (Random Forest e XGBoost) não precisam disso e recebem os
  valores originais.
- **Ordinais** (colesterol e glicose, em 3 níveis): mantidas como 1, 2 e 3, porque a ordem tem significado
  ("muito acima" é mais que "acima").
- **Binárias** (sexo, fumo, álcool e atividade física): mantidas como 0 e 1.
- **Categorias do conjunto B** (categoria de pressão e faixa de IMC): transformadas em uma coluna de sim/não
  para cada categoria (*one-hot encoding*), porque não são números. Com isso, o conjunto B passa de 16 para
  22 colunas.

Esse tratamento fica **dentro do pipeline de cada modelo**, e não é aplicado na base inteira antes da divisão.
O motivo é evitar o **vazamento de informação**: a padronização usa a média e o desvio-padrão dos dados, e,
se esses valores fossem calculados com todos os pacientes, informações da validação e do teste "vazariam"
para o treino. Dentro do pipeline, eles são calculados só com o treino (e, na validação cruzada, só com o
pedaço de treino de cada rodada) e depois apenas aplicados aos demais pacientes, exatamente como aconteceria
com um paciente novo no hospital.

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
