# Results for meta-llama/Llama-3.2-3B (nsamples=16) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | **0.3491**              | **0.3508**         | 0.3430         | 0.3439     | **0.3559**      |
| arc_easy      |     0.7449 | 0.6198                  | **0.6268**         | **0.6212**     | 0.6189     | **0.6343**      |
| boolq         |     0.7416 | **0.6773**              | 0.6172             | **0.6565**     | **0.6804** | 0.6503          |
| hellaswag     |     0.5582 | 0.5183                  | **0.5289**         | 0.5160         | **0.5184** | **0.5342**      |
| openbookqa    |     0.312  | **0.3021**              | **0.3111**         | 0.2973         | 0.3016     | **0.3114**      |
| rte           |     0.5415 | **0.5672**              | **0.5663**         | 0.5542         | 0.5478     | **0.5623**      |
| winogrande    |     0.6938 | 0.6450                  | **0.6518**         | 0.6414         | **0.6494** | **0.6593**      |
| Mean          |     0.5742 | 0.5255                  | 0.5219             | 0.5185         | 0.5229     | 0.5297          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!10} 0.3491 & \cellcolor{green!25} 0.3508 & 0.3430 & 0.3439 & \cellcolor{green!40} 0.3559 \\
arc_easy & 0.7449 & 0.6198 & \cellcolor{green!25} 0.6268 & \cellcolor{green!10} 0.6212 & 0.6189 & \cellcolor{green!40} 0.6343 \\
boolq & 0.7416 & \cellcolor{green!25} 0.6773 & 0.6172 & \cellcolor{green!10} 0.6565 & \cellcolor{green!40} 0.6804 & 0.6503 \\
hellaswag & 0.5582 & 0.5183 & \cellcolor{green!25} 0.5289 & 0.5160 & \cellcolor{green!10} 0.5184 & \cellcolor{green!40} 0.5342 \\
openbookqa & 0.3120 & \cellcolor{green!10} 0.3021 & \cellcolor{green!25} 0.3111 & 0.2973 & 0.3016 & \cellcolor{green!40} 0.3114 \\
rte & 0.5415 & \cellcolor{green!40} 0.5672 & \cellcolor{green!25} 0.5663 & 0.5542 & 0.5478 & \cellcolor{green!10} 0.5623 \\
winogrande & 0.6938 & 0.6450 & \cellcolor{green!25} 0.6518 & 0.6414 & \cellcolor{green!10} 0.6494 & \cellcolor{green!40} 0.6593 \\
Mean & 0.5742 & 0.5255 & 0.5219 & 0.5185 & 0.5229 & 0.5297 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.427474 |        0.355909 |
| arc_easy      |   0.744949 |        0.634312 |
| boolq         |   0.74159  |        0.650344 |
| hellaswag     |   0.558156 |        0.534231 |
| openbookqa    |   0.312    |        0.311375 |
| rte           |   0.541516 |        0.562274 |
| winogrande    |   0.693765 |        0.659333 |
| Mean          |   0.574207 |        0.529683 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.3559 \\
arc_easy & 0.7449 & 0.6343 \\
boolq & 0.7416 & 0.6503 \\
hellaswag & 0.5582 & 0.5342 \\
openbookqa & 0.3120 & 0.3114 \\
rte & 0.5415 & 0.5623 \\
winogrande & 0.6938 & 0.6593 \\
Mean & 0.5742 & 0.5297 \\
\bottomrule
\end{tabular}
```
