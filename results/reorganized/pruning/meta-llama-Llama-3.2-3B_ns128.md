# Results for meta-llama/Llama-3.2-3B (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | **0.3521**              | **0.3573**         | 0.3467         | 0.3510     | **0.3559**      |
| arc_easy      |     0.7449 | **0.6278**              | **0.6276**         | 0.6247         | 0.6242     | **0.6304**      |
| boolq         |     0.7416 | **0.6799**              | 0.6309             | **0.6704**     | **0.6848** | 0.6548          |
| hellaswag     |     0.5582 | 0.5206                  | **0.5295**         | 0.5202         | **0.5239** | **0.5296**      |
| openbookqa    |     0.312  | 0.2993                  | **0.3084**         | 0.3010         | **0.3047** | **0.3130**      |
| rte           |     0.5415 | 0.5542                  | **0.5718**         | 0.5537         | **0.5614** | **0.5695**      |
| winogrande    |     0.6938 | 0.6488                  | **0.6532**         | 0.6513         | **0.6551** | **0.6549**      |
| Mean          |     0.5742 | 0.5261                  | 0.5255             | 0.5240         | 0.5293     | 0.5297          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!10} 0.3521 & \cellcolor{green!40} 0.3573 & 0.3467 & 0.3510 & \cellcolor{green!25} 0.3559 \\
arc_easy & 0.7449 & \cellcolor{green!25} 0.6278 & \cellcolor{green!10} 0.6276 & 0.6247 & 0.6242 & \cellcolor{green!40} 0.6304 \\
boolq & 0.7416 & \cellcolor{green!25} 0.6799 & 0.6309 & \cellcolor{green!10} 0.6704 & \cellcolor{green!40} 0.6848 & 0.6548 \\
hellaswag & 0.5582 & 0.5206 & \cellcolor{green!25} 0.5295 & 0.5202 & \cellcolor{green!10} 0.5239 & \cellcolor{green!40} 0.5296 \\
openbookqa & 0.3120 & 0.2993 & \cellcolor{green!25} 0.3084 & 0.3010 & \cellcolor{green!10} 0.3047 & \cellcolor{green!40} 0.3130 \\
rte & 0.5415 & 0.5542 & \cellcolor{green!40} 0.5718 & 0.5537 & \cellcolor{green!10} 0.5614 & \cellcolor{green!25} 0.5695 \\
winogrande & 0.6938 & 0.6488 & \cellcolor{green!10} 0.6532 & 0.6513 & \cellcolor{green!40} 0.6551 & \cellcolor{green!25} 0.6549 \\
Mean & 0.5742 & 0.5261 & 0.5255 & 0.5240 & 0.5293 & 0.5297 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.427474 | 0.375178 |        0.355909 |
| arc_easy      |   0.744949 | 0.654242 |        0.630419 |
| boolq         |   0.74159  | 0.671885 |        0.654817 |
| hellaswag     |   0.558156 | 0.562701 |        0.529638 |
| openbookqa    |   0.312    | 0.320958 |        0.313    |
| rte           |   0.541516 | 0.536552 |        0.569495 |
| winogrande    |   0.693765 | 0.666732 |        0.654893 |
| Mean          |   0.574207 | 0.541178 |        0.529739 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{blue!15} \textbf{0.3752} & 0.3559 \\
arc_easy & 0.7449 & \cellcolor{blue!15} \textbf{0.6542} & 0.6304 \\
boolq & 0.7416 & \cellcolor{blue!15} \textbf{0.6719} & 0.6548 \\
hellaswag & 0.5582 & \cellcolor{blue!15} \textbf{0.5627} & 0.5296 \\
openbookqa & 0.3120 & \cellcolor{blue!15} \textbf{0.3210} & 0.3130 \\
rte & 0.5415 & 0.5366 & \cellcolor{blue!15} \textbf{0.5695} \\
winogrande & 0.6938 & \cellcolor{blue!15} \textbf{0.6667} & 0.6549 \\
Mean & 0.5742 & 0.5412 & 0.5297 \\
\bottomrule
\end{tabular}
```
