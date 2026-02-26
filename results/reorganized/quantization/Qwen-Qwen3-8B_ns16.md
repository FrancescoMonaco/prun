# Results for Qwen/Qwen3-8B (nsamples=16) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5276                  | **0.5371**         | **0.5330**     | **0.5309** | 0.5281          |
| arc_easy      |     0.8359 | 0.7934                  | **0.7961**         | **0.7944**     | **0.7973** | 0.7941          |
| boolq         |     0.8661 | **0.8580**              | 0.8549             | **0.8588**     | 0.8552     | **0.8591**      |
| hellaswag     |     0.5713 | **0.6445**              | 0.6414             | **0.6442**     | 0.6432     | **0.6433**      |
| openbookqa    |     0.31   | **0.3556**              | **0.3548**         | 0.3530         | 0.3523     | **0.3565**      |
| rte           |     0.7834 | 0.7721                  | **0.7798**         | **0.7793**     | **0.7735** | 0.7563          |
| winogrande    |     0.6772 | **0.6728**              | 0.6648             | **0.6697**     | 0.6670     | **0.6696**      |
| Mean          |     0.6574 | 0.6606                  | 0.6613             | 0.6618         | 0.6599     | 0.6581          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5276 & \cellcolor{green!40} 0.5371 & \cellcolor{green!25} 0.5330 & \cellcolor{green!10} 0.5309 & 0.5281 \\
arc_easy & 0.8359 & 0.7934 & \cellcolor{green!25} 0.7961 & \cellcolor{green!10} 0.7944 & \cellcolor{green!40} 0.7973 & 0.7941 \\
boolq & 0.8661 & \cellcolor{green!10} 0.8580 & 0.8549 & \cellcolor{green!25} 0.8588 & 0.8552 & \cellcolor{green!40} 0.8591 \\
hellaswag & 0.5713 & \cellcolor{green!40} 0.6445 & 0.6414 & \cellcolor{green!25} 0.6442 & 0.6432 & \cellcolor{green!10} 0.6433 \\
openbookqa & 0.3100 & \cellcolor{green!25} 0.3556 & \cellcolor{green!10} 0.3548 & 0.3530 & 0.3523 & \cellcolor{green!40} 0.3565 \\
rte & 0.7834 & 0.7721 & \cellcolor{green!40} 0.7798 & \cellcolor{green!25} 0.7793 & \cellcolor{green!10} 0.7735 & 0.7563 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6728 & 0.6648 & \cellcolor{green!25} 0.6697 & 0.6670 & \cellcolor{green!10} 0.6696 \\
Mean & 0.6574 & 0.6606 & 0.6613 & 0.6618 & 0.6599 & 0.6581 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.55802  |        0.52805  |
| arc_easy      |   0.835859 |        0.794139 |
| boolq         |   0.866055 |        0.859136 |
| hellaswag     |   0.571301 |        0.643304 |
| openbookqa    |   0.31     |        0.3565   |
| rte           |   0.783394 |        0.756318 |
| winogrande    |   0.67719  |        0.669594 |
| Mean          |   0.657403 |        0.658149 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5281 \\
arc_easy & 0.8359 & 0.7941 \\
boolq & 0.8661 & 0.8591 \\
hellaswag & 0.5713 & 0.6433 \\
openbookqa & 0.3100 & 0.3565 \\
rte & 0.7834 & 0.7563 \\
winogrande & 0.6772 & 0.6696 \\
Mean & 0.6574 & 0.6581 \\
\bottomrule
\end{tabular}
```
