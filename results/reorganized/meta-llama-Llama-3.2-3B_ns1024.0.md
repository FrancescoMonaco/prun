# Results for meta-llama/Llama-3.2-3B (nsamples=1024.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4275 | **0.4371** |
| arc_easy      |     0.7449 | **0.7213** |
| boolq         |     0.7416 | **0.7384** |
| hellaswag     |     0.5582 | **0.6414** |
| openbookqa    |     0.312  | **0.3523** |
| rte           |     0.5415 | **0.5764** |
| winogrande    |     0.6938 | **0.7001** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!40} 0.4371 \\
arc_easy & 0.7449 & \cellcolor{green!40} 0.7213 \\
boolq & 0.7416 & \cellcolor{green!40} 0.7384 \\
hellaswag & 0.5582 & \cellcolor{green!40} 0.6414 \\
openbookqa & 0.3120 & \cellcolor{green!40} 0.3523 \\
rte & 0.5415 & \cellcolor{green!40} 0.5764 \\
winogrande & 0.6938 & \cellcolor{green!40} 0.7001 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.427474 | 0.43302  |
| arc_easy      |   0.744949 | 0.7164   |
| boolq         |   0.74159  | 0.73792  |
| hellaswag     |   0.558156 | 0.640958 |
| openbookqa    |   0.312    | 0.353667 |
| rte           |   0.541516 | 0.552347 |
| winogrande    |   0.693765 | 0.704551 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4275 & 0.4330 \\
arc_easy & 0.7449 & 0.7164 \\
boolq & 0.7416 & 0.7379 \\
hellaswag & 0.5582 & 0.6410 \\
openbookqa & 0.3120 & 0.3537 \\
rte & 0.5415 & 0.5523 \\
winogrande & 0.6938 & 0.7046 \\
\bottomrule
\end{tabular}
```
