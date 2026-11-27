# DeepLearning

Shared production repository for the development of the course project for DeepLearning: **From Image Classification to Image Captioning**.

## Enunciado do projeto

### Informação geral

- **Unidade curricular:** Deep Learning (IST, 2026–27)
- **Grupos:** 3 estudantes
- **Data de entrega:** 11 de dezembro de 2026
- **Avaliação:** primeira aula prática da semana de 14–18 de dezembro de 2026
- **Entrega:** um único ficheiro ZIP no Fénix, com o relatório PDF e todo o código necessário para reproduzir os resultados.
- **Pontuação:** Parte 1 — 20 pontos; Parte 2 — 40 pontos; Parte 3 — 40 pontos; Parte 4 — até 20 pontos extra. A nota final está limitada a 100 pontos.

O objetivo é construir um sistema que descreva imagens em linguagem natural: começar com um MLP para classificação, avançar para CNNs e terminar com um descodificador RNN para geração de legendas.

O relatório deve incluir um parágrafo claro sobre a contribuição de cada membro e o checklist de colaboração e utilização de IA. Cada item em falta implica uma penalização de 10 pontos.

### Entregáveis e metodologia

- Relatório de uma coluna, usando o template LaTeX fornecido pela cadeira; máximo de 8 páginas, ou 10 páginas se a Parte 4 for tentada, sem contar referências e apêndices.
- Responder às perguntas pela ordem e usar os respetivos números, por exemplo `Q2.3`.
- Código capaz de reproduzir todas as tabelas e figuras, acompanhado por instruções de execução.
- Não incluir o dataset nem checkpoints no ZIP final.
- Reportar resultados como média ± desvio-padrão em pelo menos 3 seeds, salvo indicação em contrário.
- Todas as decisões de seleção de modelos devem usar apenas o conjunto de validação; o teste deve ser usado apenas para resultados finais.

### Dataset

Usa-se o **COCO-Dominant10-64**, com imagens RGB de `64 × 64`, 10 classes (`airplane`, `bus`, `cat`, `dog`, `elephant`, `giraffe`, `horse`, `motorcycle`, `train`, `truck`) e cinco legendas de referência por imagem.

| Split | Imagens | Imagens por classe | Legendas |
|---|---:|---:|---:|
| Train | 6 500 | 650 | 32 500 |
| Validation | 1 000 | 100 | 5 000 |
| Test | 1 000 | 100 | 5 000 |

As legendas são em inglês, têm cerca de 11 tokens em média e devem ser tratadas como dados humanos reais, incluindo possíveis erros e inconsistências. O dataset é para utilização não comercial.

### Software permitido

Nas Partes 2 e 3 podem ser usados Python, PyTorch, `torchvision.transforms` apenas para resizing/augmentations, `matplotlib`, `tqdm` e o código fornecido. São permitidas camadas PyTorch como `Linear`, `Conv2d`, `Embedding`, `GRU` e `LSTM`.

Não são permitidos `numpy`, `scikit-learn`, frameworks de treino de alto nível, `torchvision.models`, `timm`, `transformers` da Hugging Face, pesos pré-treinados ou tokenizers pré-treinados. Todo o código adicional deve ser escrito em PyTorch simples. A Parte 4 relaxa estas restrições conforme indicado abaixo.

### Parte 1 — Questões teóricas (20 pontos)

#### 1.1 Convoluções como camadas lineares condicionadas

Para um sinal 1D com padding circular e um kernel de tamanho ímpar:

- **1.1(a), 3 pontos:** mostrar que a convolução pode ser escrita como `y = W x`, descrever a estrutura de `W` e comparar os parâmetros livres com uma camada densa.
- **1.1(b), 4 pontos:** provar equivariância a translações, preservar a propriedade após ReLU, deduzir a invariância do global average pooling e mostrar que uma camada densa é equivariante se e só se a matriz for circulante.
- **1.1(c), 3 pontos:** contar pesos e biases de uma camada densa e de uma convolução `3 × 3` com 16 canais para uma imagem RGB `64 × 64`; discutir ganhos, perdas e relação com a Q2.5.

#### 1.2 Gradientes através do tempo

Para uma RNN vanilla `h_t = tanh(W h_{t-1} + U x_t + b)`:

- **1.2(a), 3 pontos:** derivar o Jacobiano entre estados consecutivos e escrevê-lo como produto para estados separados por vários passos.
- **1.2(b), 4 pontos:** limitar a norma do Jacobiano por `σ_max(W)^(T-k)`, explicar gradientes evanescentes/explosivos e justificar por que gradient clipping ajuda contra explosão mas não contra desaparecimento.
- **1.2(c), 3 pontos:** derivar o gradiente do estado de célula de uma LSTM, explicar a redução do problema dos gradientes evanescentes e o uso de bias positivo na forget gate.

### Parte 2 — Classificação de imagens com MLPs e CNNs (40 pontos)

- **2.1, 5 pontos — Baseline linear:** carregar e normalizar com estatísticas por canal calculadas apenas no treino; reportar splits, distribuição de classes e estatísticas; treinar regressão logística com SGD; mostrar loss de treino, accuracy de validação e accuracy de teste do melhor checkpoint.
- **2.2, 10 pontos — MLP:** implementar profundidade, largura, ativação e dropout configuráveis; selecionar optimizer, schedule, epochs e early stopping através da validação; reportar configuração, pesquisa, curvas e test accuracy como média ± desvio-padrão em 3 seeds.
- **2.3, 10 pontos — Estudo rigoroso:** variar pelo menos três decisões de desenho, uma de cada vez, com o mesmo orçamento; usar 3 seeds; apresentar tabelas, underfitting/overfitting, accuracies por classe e matriz de confusão.
- **2.4, 7 pontos — Bias indutivo:** visualizar pesos da primeira camada e templates da baseline; avaliar imagens com shifts de 1, 2, … pixels; aplicar uma permutação fixa de pixels a todos os splits, retreinar e interpretar os resultados.
- **2.5, 8 pontos — CNN:** implementar uma CNN com número de parâmetros comparável ou inferior ao melhor MLP; comparar baseline, MLP e CNN; repetir experiências de shift e permutação e explicar os resultados com base na Q1.1.

### Parte 3 — Image captioning com descodificador RNN (40 pontos)

- **3.1, 5 pontos — Pré-processamento:** criar tokenizer próprio, vocabulário apenas com captions de treino, `<unk>`, `<bos>`, `<eos>` e `<pad>`; implementar `encode`; reportar tamanho do vocabulário, percentagem de `<unk>` na validação e distribuição de comprimentos; ignorar padding na loss.
- **3.2, 15 pontos — Captioner RNN:** implementar encoder de imagem MLP ou CNN e descodificador GRU/LSTM com embeddings; treinar com teacher forcing e cross-entropy por token; gerar captions por greedy decoding; reportar curvas de loss/perplexity, BLEU-4 e pelo menos 8 exemplos de teste, incluindo bons e maus casos.
- **3.3(a), 7 pontos — Análise:** comparar encoder MLP/CNN e encoder congelado/conjunto; reportar métricas e número de parâmetros como média ± desvio-padrão em 3 seeds.
- **3.3(b), 5 pontos — Classe mencionada:** medir a frequência com que a caption menciona a classe correta; comparar com accuracy de classificação e BLEU-4.
- **3.4(a), 5 pontos — Decoding:** implementar beam search para beams 1, 3 e 5; comparar ranking por log-probabilidade e por comprimento; reportar BLEU-4, comprimento médio e diversidade de palavras/bigramas.
- **3.4(b), 3 pontos — Análise de probabilidade:** dar um exemplo em que greedy decoding não encontra a caption mais provável; provar que a probabilidade de um prefixo não aumenta ao ser estendido e explicar o efeito de length normalization.

### Parte 4 — Bonus (até 20 pontos extra)

Só deve ser tentada depois das Partes 1–3 estarem concluídas.

- **4.1, 10 pontos:** substituir o RNN por um pequeno Transformer decoder treinado de raiz, com causal self-attention e representação da imagem; implementar os blocos, sem `nn.TransformerDecoderLayer`; comparar métricas, parâmetros e tempo de treino.
- **4.2, 10 pontos:** usar encoders/decoders pré-treinados, comparar com os modelos treinados de raiz e discutir ganhos e custo computacional. Modelos de captioning prontos podem ser apenas referência.

### Código fornecido

O skeleton em [`src/`](src/) fornece `load_data.py`, `data.py`, `train.py`, `metrics.py`, `requirements.txt` e instruções de utilização. É permitido modificar estes ficheiros, mas todas as alterações devem ser descritas no relatório.

O enunciado original em PDF está em [`project.pdf`](project.pdf).

### Template do relatório

O template LaTeX oficial foi obtido a partir da página da cadeira e está em [`report/template/`](report/template/). O ficheiro principal é [`main.tex`](report/template/main.tex); deve ser mantido o tamanho de letra, margens e espaçamento definidos pela cadeira.
