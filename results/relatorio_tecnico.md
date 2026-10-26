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

Foram avaliados cinco algoritmos de aprendizado de máquina para identificar padrões associados à presença de doenças cardiovasculares. A comparação considerou desempenho preditivo, capacidade de identificar pacientes com a doença, interpretabilidade e custo computacional. Como o objetivo é apoiar a triagem, a sensibilidade (*recall*) recebeu atenção especial: um falso negativo representa um paciente com doença que não foi sinalizado pelo modelo.

**Regressão Logística (T15).** Foi utilizada como modelo de referência (*baseline*) por sua simplicidade e interpretabilidade. Seus coeficientes permitem compreender a associação de cada variável com a classificação, por meio de medidas como o *odds ratio*. Foram avaliados diferentes valores de regularização (`C`) e o balanceamento das classes. A configuração definida na T15 foi `C=1` e `class_weight="balanced"`, buscando preservar a capacidade de identificar pacientes positivos.

**K-Nearest Neighbors — KNN (T16).** Classifica um paciente considerando os resultados dos pacientes mais semelhantes presentes no treinamento. Sua inclusão permitiu avaliar uma abordagem baseada em distância, diferente dos modelos lineares e das árvores. Como a distância depende da escala das variáveis, medidas contínuas foram padronizadas. A configuração escolhida utiliza 101 vizinhos com pesos uniformes. Embora o treinamento seja rápido, a previsão exige comparar novos pacientes com os registros armazenados.

**Random Forest (T17).** Combina 300 árvores de decisão para identificar relações complexas entre fatores como idade, pressão arterial e colesterol. Foi escolhido pela capacidade de representar associações não lineares e reduzir a instabilidade de uma árvore isolada. Os experimentos iniciais apresentaram sobreajuste, posteriormente reduzido por limitações na profundidade das árvores e na quantidade mínima de pacientes por folha.

**XGBoost (T18).** Utiliza árvores construídas sequencialmente, de modo que cada nova árvore procure corrigir erros das anteriores. Foi incluído por sua eficiência em dados tabulares e pela capacidade de representar interações entre fatores clínicos. Para controlar o sobreajuste, foram utilizadas árvores de profundidade 4 e taxa de aprendizado de 0,03. O conjunto A apresentou AUC ROC de aproximadamente 0,792 na validação e foi selecionado como modelo final do projeto.

**Support Vector Machine — SVM (T19).** Busca uma fronteira que diferencie pacientes com e sem doença cardiovascular. Para representar relações não lineares sem o elevado custo do kernel RBF exato, foi utilizada a aproximação Nyström, com 500 componentes, seguida de `LinearSVC` calibrado. Essa estratégia apresentou AUC próxima de 0,792 na validação e permitiu aproveitar o conjunto completo de treinamento com menor custo computacional.

### Conjuntos de atributos e engenharia de variáveis

Todos os algoritmos foram avaliados com dois conjuntos de atributos. O **conjunto A** contém 11 variáveis clínicas, demográficas e comportamentais, incluindo idade, sexo, medidas corporais, pressão arterial, colesterol, glicose e hábitos de vida. O **conjunto B** acrescenta cinco variáveis derivadas: índice de massa corporal (IMC), pressão de pulso, pressão arterial média (PAM), categoria de pressão e faixa de IMC.

A comparação mostrou diferenças pequenas de AUC ROC entre os conjuntos A e B, inferiores a 0,01. Portanto, a engenharia de variáveis não produziu ganho expressivo na capacidade de discriminação dos modelos. O conjunto A foi preferido para o modelo final por apresentar desempenho competitivo com menor complexidade.

### Busca de hiperparâmetros e custo computacional

A otimização utilizou `RandomizedSearchCV`, com validação cruzada estratificada em cinco partes, exclusivamente nos dados de treinamento. O principal critério de escolha foi a AUC ROC, que avalia a capacidade de distinguir as duas classes em diferentes limiares de decisão. O pré-processamento foi executado dentro de cada etapa da validação cruzada, evitando vazamento de informações.

Para o KNN, a busca utilizou uma amostra estratificada de 10.000 pacientes, reduzindo o custo das comparações de distância. Para o SVM, foram testadas abordagens lineares, kernel RBF exato em amostra e aproximação Nyström; esta última foi selecionada pelo equilíbrio entre desempenho e tempo de processamento.

| Modelo | Hiperparâmetros principais | Tempo registrado |
|---|---|---|
| Regressão Logística | `C=1`; `class_weight=balanced` | 0,21 s* |
| KNN | `n_neighbors=101`; `weights=uniform` | 0,13 s |
| Random Forest A | `n_estimators=300`; `max_depth=10`; `min_samples_leaf=20`; `max_features=sqrt` | 216,76 s |
| XGBoost A | `n_estimators=200`; `max_depth=4`; `learning_rate=0.03`; `subsample=0.8`; `colsample_bytree=0.8` | 0,52 s |
| SVM A | `C=0.1`; `gamma=0.1`; Nyström com 500 componentes | 29,20 s |

*Nota:* os tempos foram obtidos dos registros disponíveis, mas representam etapas distintas: no Random Forest, o valor inclui a busca de hiperparâmetros; nos demais casos, os registros apresentados correspondem a execuções de treinamento. O tempo da Regressão Logística ainda precisa ser confirmado para a configuração balanceada.

Os resultados comparativos podem ser consultados em `results/comparison_table.csv`, e as curvas ROC dos modelos estão disponíveis em `results/figures/roc_A.png` e `results/figures/roc_A_vs_B_xgb.png`.

**Síntese.** Os cinco modelos apresentaram capacidade discriminativa semelhante, mas diferiram em interpretabilidade, custo computacional e comportamento de classificação. O XGBoost com o conjunto A foi selecionado para a etapa final, com ajuste posterior do limiar de decisão para priorizar a identificação de pacientes com doença cardiovascular. Os resultados representam validação experimental interna e não autorizam o uso isolado do sistema para diagnóstico médico.

## 6. Resultados
## 7. Interpretação (feature importance / SHAP)
## 8. Discussão crítica (uso na prática — o médico tem a palavra final)
## 9. Conclusão
