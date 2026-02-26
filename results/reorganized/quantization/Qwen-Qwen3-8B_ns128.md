# Results for Qwen/Qwen3-8B (nsamples=128) - quantization

## Average across Calibration Groups

| task          |   original | random     |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.558  | **0.5337** |
| arc_easy      |     0.8359 | **0.8009** |
| boolq         |     0.8661 | **0.8633** |
| hellaswag     |     0.5713 | **0.6472** |
| openbookqa    |     0.31   | **0.3690** |
| rte           |     0.7834 | **0.7906** |
| winogrande    |     0.6772 | **0.6835** |
| Mean          |     0.6574 | 0.6698     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & random \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!40} 0.5337 \\
arc_easy & 0.8359 & \cellcolor{green!40} 0.8009 \\
boolq & 0.8661 & \cellcolor{green!40} 0.8633 \\
hellaswag & 0.5713 & \cellcolor{green!40} 0.6472 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3690 \\
rte & 0.7834 & \cellcolor{green!40} 0.7906 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6835 \\
Mean & 0.6574 & 0.6698 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.55802  | 0.552474 |
| arc_easy      |   0.835859 | 0.819602 |
| boolq         |   0.866055 | 0.862462 |
| hellaswag     |   0.571301 | 0.649746 |
| openbookqa    |   0.31     | 0.36525  |
| rte           |   0.783394 | 0.767148 |
| winogrande    |   0.67719  | 0.677979 |
| Mean          |   0.657403 | 0.670666 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.5580 & 0.5525 \\
arc_easy & 0.8359 & 0.8196 \\
boolq & 0.8661 & 0.8625 \\
hellaswag & 0.5713 & 0.6497 \\
openbookqa & 0.3100 & 0.3652 \\
rte & 0.7834 & 0.7671 \\
winogrande & 0.6772 & 0.6780 \\
Mean & 0.6574 & 0.6707 \\
\bottomrule
\end{tabular}
```
