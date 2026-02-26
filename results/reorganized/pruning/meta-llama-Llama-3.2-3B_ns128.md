# Results for meta-llama/Llama-3.2-3B (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | 0.3549                  | **0.3643**         | **0.3656**     | **0.3609** | 0.3605          |
| arc_easy      |     0.7449 | 0.6326                  | 0.6305             | **0.6441**     | **0.6420** | **0.6330**      |
| boolq         |     0.7416 | **0.6927**              | 0.6480             | **0.6954**     | 0.6847     | **0.6924**      |
| hellaswag     |     0.5582 | 0.5265                  | **0.5364**         | 0.5272         | **0.5314** | **0.5381**      |
| openbookqa    |     0.312  | **0.3020**              | **0.3100**         | **0.3090**     | 0.3010     | 0.3010          |
| rte           |     0.5415 | **0.5668**              | **0.6065**         | 0.5668         | 0.5523     | **0.5740**      |
| winogrande    |     0.6938 | 0.6511                  | **0.6582**         | 0.6527         | **0.6567** | **0.6598**      |
| Mean          |     0.5742 | 0.5324                  | 0.5363             | 0.5373         | 0.5327     | 0.5370          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.3549 & \cellcolor{green!25} 0.3643 & \cellcolor{green!40} 0.3656 & \cellcolor{green!10} 0.3609 & 0.3605 \\
arc_easy & 0.7449 & 0.6326 & 0.6305 & \cellcolor{green!40} 0.6441 & \cellcolor{green!25} 0.6420 & \cellcolor{green!10} 0.6330 \\
boolq & 0.7416 & \cellcolor{green!25} 0.6927 & 0.6480 & \cellcolor{green!40} 0.6954 & 0.6847 & \cellcolor{green!10} 0.6924 \\
hellaswag & 0.5582 & 0.5265 & \cellcolor{green!25} 0.5364 & 0.5272 & \cellcolor{green!10} 0.5314 & \cellcolor{green!40} 0.5381 \\
openbookqa & 0.3120 & \cellcolor{green!10} 0.3020 & \cellcolor{green!40} 0.3100 & \cellcolor{green!25} 0.3090 & 0.3010 & 0.3010 \\
rte & 0.5415 & \cellcolor{green!10} 0.5668 & \cellcolor{green!40} 0.6065 & 0.5668 & 0.5523 & \cellcolor{green!25} 0.5740 \\
winogrande & 0.6938 & 0.6511 & \cellcolor{green!25} 0.6582 & 0.6527 & \cellcolor{green!10} 0.6567 & \cellcolor{green!40} 0.6598 \\
Mean & 0.5742 & 0.5324 & 0.5363 & 0.5373 & 0.5327 & 0.5370 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.427474 | 0.430887 |        0.360495 |
| arc_easy      |   0.744949 | 0.719276 |        0.632997 |
| boolq         |   0.74159  | 0.732951 |        0.692355 |
| hellaswag     |   0.558156 | 0.641132 |        0.53809  |
| openbookqa    |   0.312    | 0.352    |        0.301    |
| rte           |   0.541516 | 0.537004 |        0.574007 |
| winogrande    |   0.693765 | 0.702447 |        0.659826 |
| Mean          |   0.574207 | 0.587957 |        0.536967 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{blue!15} \textbf{0.4309} & 0.3605 \\
arc_easy & 0.7449 & \cellcolor{blue!15} \textbf{0.7193} & 0.6330 \\
boolq & 0.7416 & \cellcolor{blue!15} \textbf{0.7330} & 0.6924 \\
hellaswag & 0.5582 & \cellcolor{blue!15} \textbf{0.6411} & 0.5381 \\
openbookqa & 0.3120 & \cellcolor{blue!15} \textbf{0.3520} & 0.3010 \\
rte & 0.5415 & 0.5370 & \cellcolor{blue!15} \textbf{0.5740} \\
winogrande & 0.6938 & \cellcolor{blue!15} \textbf{0.7024} & 0.6598 \\
Mean & 0.5742 & 0.5880 & 0.5370 \\
\bottomrule
\end{tabular}
```
