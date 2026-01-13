# Results for Qwen/Qwen3-8B (nsamples=1024.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.558  | **0.5595** |
| arc_easy      |     0.8359 | **0.8126** |
| boolq         |     0.8661 | **0.8651** |
| hellaswag     |     0.5713 | **0.6531** |
| openbookqa    |     0.31   | **0.3700** |
| rte           |     0.7834 | **0.7635** |
| winogrande    |     0.6772 | **0.6942** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!40} 0.5595 \\
arc_easy & 0.8359 & \cellcolor{green!40} 0.8126 \\
boolq & 0.8661 & \cellcolor{green!40} 0.8651 \\
hellaswag & 0.5713 & \cellcolor{green!40} 0.6531 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3700 \\
rte & 0.7834 & \cellcolor{green!40} 0.7635 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6942 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.55802  | 0.554892 |
| arc_easy      |   0.835859 | 0.820497 |
| boolq         |   0.866055 | 0.862181 |
| hellaswag     |   0.571301 | 0.649539 |
| openbookqa    |   0.31     | 0.364667 |
| rte           |   0.783394 | 0.766546 |
| winogrande    |   0.67719  | 0.684294 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.5580 & 0.5549 \\
arc_easy & 0.8359 & 0.8205 \\
boolq & 0.8661 & 0.8622 \\
hellaswag & 0.5713 & 0.6495 \\
openbookqa & 0.3100 & 0.3647 \\
rte & 0.7834 & 0.7665 \\
winogrande & 0.6772 & 0.6843 \\
\bottomrule
\end{tabular}
```
