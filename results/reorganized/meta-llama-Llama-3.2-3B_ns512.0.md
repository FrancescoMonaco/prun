# Results for meta-llama/Llama-3.2-3B (nsamples=512.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4275 | **0.4387** |
| arc_easy      |     0.7449 | **0.7196** |
| boolq         |     0.7416 | **0.7379** |
| hellaswag     |     0.5582 | **0.6421** |
| openbookqa    |     0.312  | **0.3487** |
| rte           |     0.5415 | **0.5716** |
| winogrande    |     0.6938 | **0.7019** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!40} 0.4387 \\
arc_easy & 0.7449 & \cellcolor{green!40} 0.7196 \\
boolq & 0.7416 & \cellcolor{green!40} 0.7379 \\
hellaswag & 0.5582 & \cellcolor{green!40} 0.6421 \\
openbookqa & 0.3120 & \cellcolor{green!40} 0.3487 \\
rte & 0.5415 & \cellcolor{green!40} 0.5716 \\
winogrande & 0.6938 & \cellcolor{green!40} 0.7019 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.427474 | 0.431172 |
| arc_easy      |   0.744949 | 0.718013 |
| boolq         |   0.74159  | 0.737819 |
| hellaswag     |   0.558156 | 0.641091 |
| openbookqa    |   0.312    | 0.352333 |
| rte           |   0.541516 | 0.540313 |
| winogrande    |   0.693765 | 0.700605 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4275 & 0.4312 \\
arc_easy & 0.7449 & 0.7180 \\
boolq & 0.7416 & 0.7378 \\
hellaswag & 0.5582 & 0.6411 \\
openbookqa & 0.3120 & 0.3523 \\
rte & 0.5415 & 0.5403 \\
winogrande & 0.6938 & 0.7006 \\
\bottomrule
\end{tabular}
```
