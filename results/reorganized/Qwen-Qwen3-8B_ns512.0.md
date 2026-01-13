# Results for Qwen/Qwen3-8B (nsamples=512.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.558  | **0.5586** |
| arc_easy      |     0.8359 | **0.8167** |
| boolq         |     0.8661 | **0.8655** |
| hellaswag     |     0.5713 | **0.6537** |
| openbookqa    |     0.31   | **0.3693** |
| rte           |     0.7834 | **0.7605** |
| winogrande    |     0.6772 | **0.6848** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!40} 0.5586 \\
arc_easy & 0.8359 & \cellcolor{green!40} 0.8167 \\
boolq & 0.8661 & \cellcolor{green!40} 0.8655 \\
hellaswag & 0.5713 & \cellcolor{green!40} 0.6537 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3693 \\
rte & 0.7834 & \cellcolor{green!40} 0.7605 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6848 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.55802  | 0.553328 |
| arc_easy      |   0.835859 | 0.819935 |
| boolq         |   0.866055 | 0.863303 |
| hellaswag     |   0.571301 | 0.649622 |
| openbookqa    |   0.31     | 0.363667 |
| rte           |   0.783394 | 0.766546 |
| winogrande    |   0.67719  | 0.68061  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.5580 & 0.5533 \\
arc_easy & 0.8359 & 0.8199 \\
boolq & 0.8661 & 0.8633 \\
hellaswag & 0.5713 & 0.6496 \\
openbookqa & 0.3100 & 0.3637 \\
rte & 0.7834 & 0.7665 \\
winogrande & 0.6772 & 0.6806 \\
\bottomrule
\end{tabular}
```
