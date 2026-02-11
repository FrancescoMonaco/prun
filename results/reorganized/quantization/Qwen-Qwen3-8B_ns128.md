# Results for Qwen/Qwen3-8B (nsamples=128) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5323                  | **0.5378**         | 0.5327         | **0.5328** | **0.5462**      |
| arc_easy      |     0.8359 | **0.8011**              | **0.8027**         | 0.7986         | 0.7904     | **0.8026**      |
| boolq         |     0.8661 | **0.8589**              | 0.8560             | **0.8597**     | 0.8547     | **0.8613**      |
| hellaswag     |     0.5713 | 0.6451                  | **0.6456**         | 0.6453         | **0.6463** | **0.6467**      |
| openbookqa    |     0.31   | 0.3529                  | **0.3581**         | **0.3570**     | 0.3552     | **0.3599**      |
| rte           |     0.7834 | **0.7820**              | **0.7730**         | 0.7671         | **0.7735** | 0.7662          |
| winogrande    |     0.6772 | 0.6776                  | **0.6777**         | **0.6785**     | 0.6711     | **0.6796**      |
| Mean          |     0.6574 | 0.6643                  | 0.6644             | 0.6627         | 0.6606     | 0.6661          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5323 & \cellcolor{green!25} 0.5378 & 0.5327 & \cellcolor{green!10} 0.5328 & \cellcolor{green!40} 0.5462 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.8011 & \cellcolor{green!40} 0.8027 & 0.7986 & 0.7904 & \cellcolor{green!25} 0.8026 \\
boolq & 0.8661 & \cellcolor{green!10} 0.8589 & 0.8560 & \cellcolor{green!25} 0.8597 & 0.8547 & \cellcolor{green!40} 0.8613 \\
hellaswag & 0.5713 & 0.6451 & \cellcolor{green!10} 0.6456 & 0.6453 & \cellcolor{green!25} 0.6463 & \cellcolor{green!40} 0.6467 \\
openbookqa & 0.3100 & 0.3529 & \cellcolor{green!25} 0.3581 & \cellcolor{green!10} 0.3570 & 0.3552 & \cellcolor{green!40} 0.3599 \\
rte & 0.7834 & \cellcolor{green!40} 0.7820 & \cellcolor{green!10} 0.7730 & 0.7671 & \cellcolor{green!25} 0.7735 & 0.7662 \\
winogrande & 0.6772 & 0.6776 & \cellcolor{green!10} 0.6777 & \cellcolor{green!25} 0.6785 & 0.6711 & \cellcolor{green!40} 0.6796 \\
Mean & 0.6574 & 0.6643 & 0.6644 & 0.6627 & 0.6606 & 0.6661 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.55802  | 0.520158 |        0.546235 |
| arc_easy      |   0.835859 | 0.78805  |        0.802636 |
| boolq         |   0.866055 | 0.848567 |        0.861277 |
| hellaswag     |   0.571301 | 0.609157 |        0.646665 |
| openbookqa    |   0.31     | 0.346375 |        0.359875 |
| rte           |   0.783394 | 0.738267 |        0.766245 |
| winogrande    |   0.67719  | 0.673441 |        0.679558 |
| Mean          |   0.657403 | 0.646288 |        0.66607  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5202 & \cellcolor{blue!15} \textbf{0.5462} \\
arc_easy & 0.8359 & 0.7880 & \cellcolor{blue!15} \textbf{0.8026} \\
boolq & 0.8661 & 0.8486 & \cellcolor{blue!15} \textbf{0.8613} \\
hellaswag & 0.5713 & 0.6092 & \cellcolor{blue!15} \textbf{0.6467} \\
openbookqa & 0.3100 & 0.3464 & \cellcolor{blue!15} \textbf{0.3599} \\
rte & 0.7834 & 0.7383 & \cellcolor{blue!15} \textbf{0.7662} \\
winogrande & 0.6772 & 0.6734 & \cellcolor{blue!15} \textbf{0.6796} \\
Mean & 0.6574 & 0.6463 & 0.6661 \\
\bottomrule
\end{tabular}
```
