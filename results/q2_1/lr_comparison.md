# Q2.1 — Experiência com learning rate

Mantivemos arquitetura, batch size (128), número máximo de épocas (25), normalização, SGD e seeds (0, 1, 2). Alterámos apenas o learning rate. Em cada seed foi selecionado o checkpoint com melhor accuracy de validação; o teste foi consultado depois dessa seleção.

| Learning rate | Validação, média ± desvio-padrão | Teste, média ± desvio-padrão |
|---:|---:|---:|
| 0.1 | 24.20% ± 0.50 pp | 20.83% ± 0.87 pp |
| 0.001 | 32.10% ± 0.10 pp | 26.73% ± 0.91 pp |

Com `0.001`, a accuracy média aumentou **7.90 pontos percentuais** na validação e **5.90 pontos percentuais** no teste. Nas três execuções, os melhores epochs ocorreram em 14, 6 e 14, respetivamente. O resultado observado favorece `0.001` para a configuração atual. Ainda assim, a conclusão é limitada a estes dois valores, três seeds e 25 épocas; não é uma procura exaustiva do learning rate.

## Curvas

O gráfico por seed está em [`learning_curves.png`](../q2_1_lr0001/learning_curves.png). Os resultados completos estão em [`summary.json`](../q2_1_lr0001/summary.json). O script agora aceita `--learning-rate` e `--output-suffix`.

## Decisão pendente

Esta comparação não altera automaticamente o resultado/base aprovada da Q2.1. Se validarmos a experiência, podemos substituir a baseline registada por `0.001`, preservando esta comparação como evidência da escolha.
