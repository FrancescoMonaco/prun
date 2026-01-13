# Results for meta-llama/Llama-3.2-3B (nsamples=128.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4275 | **0.4420** |
| arc_easy      |     0.7449 | **0.7236** |
| boolq         |     0.7416 | **0.7367** |
| hellaswag     |     0.5582 | **0.6408** |
| openbookqa    |     0.312  | **0.3487** |
| rte           |     0.5415 | **0.5692** |
| winogrande    |     0.6938 | **0.7030** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!40} 0.4420 \\
arc_easy & 0.7449 & \cellcolor{green!40} 0.7236 \\
boolq & 0.7416 & \cellcolor{green!40} 0.7367 \\
hellaswag & 0.5582 & \cellcolor{green!40} 0.6408 \\
openbookqa & 0.3120 & \cellcolor{green!40} 0.3487 \\
rte & 0.5415 & \cellcolor{green!40} 0.5692 \\
winogrande & 0.6938 & \cellcolor{green!40} 0.7030 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.427474 | 0.431456 |
| arc_easy      |   0.744949 | 0.718575 |
| boolq         |   0.74159  | 0.735474 |
| hellaswag     |   0.558156 | 0.64119  |
| openbookqa    |   0.312    | 0.352667 |
| rte           |   0.541516 | 0.536703 |
| winogrande    |   0.693765 | 0.702184 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4275 & 0.4315 \\
arc_easy & 0.7449 & 0.7186 \\
boolq & 0.7416 & 0.7355 \\
hellaswag & 0.5582 & 0.6412 \\
openbookqa & 0.3120 & 0.3527 \\
rte & 0.5415 & 0.5367 \\
winogrande & 0.6938 & 0.7022 \\
\bottomrule
\end{tabular}
```
