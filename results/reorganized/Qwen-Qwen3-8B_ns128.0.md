# Results for Qwen/Qwen3-8B (nsamples=128.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.558  | **0.5545** |
| arc_easy      |     0.8359 | **0.8139** |
| boolq         |     0.8661 | **0.8661** |
| hellaswag     |     0.5713 | **0.6531** |
| openbookqa    |     0.31   | **0.3670** |
| rte           |     0.7834 | **0.7617** |
| winogrande    |     0.6772 | **0.6859** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!40} 0.5545 \\
arc_easy & 0.8359 & \cellcolor{green!40} 0.8139 \\
boolq & 0.8661 & \cellcolor{green!40} 0.8661 \\
hellaswag & 0.5713 & \cellcolor{green!40} 0.6531 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3670 \\
rte & 0.7834 & \cellcolor{green!40} 0.7617 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6859 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.55802  | 0.553043 |
| arc_easy      |   0.835859 | 0.818673 |
| boolq         |   0.866055 | 0.862691 |
| hellaswag     |   0.571301 | 0.649622 |
| openbookqa    |   0.31     | 0.365333 |
| rte           |   0.783394 | 0.766546 |
| winogrande    |   0.67719  | 0.677979 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.5580 & 0.5530 \\
arc_easy & 0.8359 & 0.8187 \\
boolq & 0.8661 & 0.8627 \\
hellaswag & 0.5713 & 0.6496 \\
openbookqa & 0.3100 & 0.3653 \\
rte & 0.7834 & 0.7665 \\
winogrande & 0.6772 & 0.6780 \\
\bottomrule
\end{tabular}
```
