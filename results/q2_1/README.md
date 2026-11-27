# Q2.1 — Baseline linear

## Estado

Esta nota regista a configuração e os resultados atuais da regressão logística multinomial. Os valores vêm de `summary.json`; a execução é reproduzível com `python src/run_q2_1.py` a partir da raiz do repositório.

## Dados e pré-processamento

| Split | Imagens | Distribuição por classe |
|---|---:|---:|
| Treino | 6 500 | 650 em cada uma das 10 classes |
| Validação | 1 000 | 100 em cada classe |
| Teste | 1 000 | 100 em cada classe |

As imagens RGB de `64 × 64` são achatadas em vetores de 12 288 valores. Os canais são normalizados com média e desvio-padrão calculados exclusivamente no conjunto de treino:

| Canal | Média | Desvio-padrão |
|---|---:|---:|
| R | 0.46338 | 0.25846 |
| G | 0.45295 | 0.25710 |
| B | 0.41729 | 0.27130 |

## Modelo e treino

O classificador é uma camada linear `Linear(12 288, 10)`, equivalente a regressão logística multinomial, com **122 890 parâmetros treináveis** (122 880 pesos e 10 biases). Foi treinado com cross-entropy, SGD, learning rate `0.1`, batch size `128` e 25 épocas. Seeds: `0`, `1` e `2`. Em cada execução, o checkpoint escolhido é o da melhor accuracy de validação; o conjunto de teste só é avaliado depois dessa seleção.

## Resultados atuais

Accuracy reportada em percentagem; média e desvio-padrão amostral entre as três seeds.

| Seed | Melhor época | Validação no melhor checkpoint | Teste no melhor checkpoint |
|---:|---:|---:|---:|
| 0 | 21 | 24.20% | 20.60% |
| 1 | 18 | 24.70% | 21.80% |
| 2 | 9 | 23.70% | 20.10% |
| **Média ± desvio-padrão** | — | **24.20% ± 0.50 pp** | **20.83% ± 0.87 pp** |

O desempenho no teste fica acima da referência uniforme (`10%` para 10 classes). A loss de treino diminui substancialmente enquanto a accuracy de validação oscila sem tendência consistente; neste protocolo, a redução da loss não se refletiu numa melhoria consistente da accuracy de validação. A variação entre estas três seeds é descritivamente visível, pelo que os resultados posteriores devem usar o mesmo protocolo. A hipótese de que uma camada linear não representa adequadamente a estrutura espacial e relações não lineares será investigada com MLPs e CNNs; estes resultados, isoladamente, não demonstram a causa do desempenho.

## Curvas

![Curvas de treino e validação da baseline linear para as três seeds](learning_curves.png)

A figura mostra os históricos de treino e validação das três seeds. A loss por época é a média simples das médias por batch do loop fornecido, pelo que o último batch, menor, tem o mesmo peso que os restantes; por isso, deve ser interpretada como uma aproximação da loss média por época.

## Texto de trabalho para o relatório

> O conjunto contém 6 500 imagens de treino e 1 000 imagens em cada um dos conjuntos de validação e teste, com distribuição equilibrada de 650 imagens de treino e 100 de validação/teste por classe. Cada imagem tem cinco legendas. Normalizámos cada canal usando a média e o desvio-padrão do treino (RGB: médias `[0.4634, 0.4530, 0.4173]`; desvios-padrão `[0.2585, 0.2571, 0.2713]`). A baseline é uma regressão logística multinomial sobre os 12 288 píxeis de cada imagem, com 122 890 parâmetros, treinada com SGD (`lr=0.1`, batch size 128) durante 25 épocas. Selecionámos separadamente por seed o checkpoint com melhor accuracy de validação. Em três seeds, obtivemos `24.20 ± 0.50%` de accuracy de validação e `20.83 ± 0.87%` no teste. A loss de treino diminui, enquanto a accuracy de validação oscila sem tendência consistente. A hipótese de que modelos com camadas ocultas ou convolucionais aproveitem melhor as relações entre píxeis será testada nas questões seguintes.

Os resultados são exploratórios e devem ser substituídos caso se altere o protocolo ou a implementação.

## Artefactos e fontes

- Resultados completos por seed: [`summary.json`](summary.json).
- Script: [`src/run_q2_1.py`](../../src/run_q2_1.py).
- Enunciado: Q2.1 em `project.pdf`, pp. 6–7.
