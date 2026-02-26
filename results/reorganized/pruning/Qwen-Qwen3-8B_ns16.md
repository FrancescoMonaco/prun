# Results for Qwen/Qwen3-8B (nsamples=16) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | **0.4969**              | **0.5056**         | 0.4937         | 0.4946     | **0.5087**      |
| arc_easy      |     0.8359 | **0.7782**              | **0.7798**         | 0.7740         | 0.7702     | **0.7819**      |
| boolq         |     0.8661 | 0.8408                  | **0.8467**         | 0.8385         | **0.8446** | **0.8436**      |
| hellaswag     |     0.5713 | **0.5738**              | **0.5776**         | 0.5717         | 0.5718     | **0.5833**      |
| openbookqa    |     0.31   | 0.3274                  | **0.3364**         | 0.3289         | **0.3306** | **0.3399**      |
| rte           |     0.7834 | **0.7243**              | **0.7220**         | **0.7283**     | 0.7157     | 0.7125          |
| winogrande    |     0.6772 | 0.6686                  | **0.6738**         | 0.6668         | **0.6700** | **0.6717**      |
| Mean          |     0.6574 | 0.6300                  | 0.6346             | 0.6288         | 0.6282     | 0.6345          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!10} 0.4969 & \cellcolor{green!25} 0.5056 & 0.4937 & 0.4946 & \cellcolor{green!40} 0.5087 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.7782 & \cellcolor{green!25} 0.7798 & 0.7740 & 0.7702 & \cellcolor{green!40} 0.7819 \\
boolq & 0.8661 & 0.8408 & \cellcolor{green!40} 0.8467 & 0.8385 & \cellcolor{green!25} 0.8446 & \cellcolor{green!10} 0.8436 \\
hellaswag & 0.5713 & \cellcolor{green!10} 0.5738 & \cellcolor{green!25} 0.5776 & 0.5717 & 0.5718 & \cellcolor{green!40} 0.5833 \\
openbookqa & 0.3100 & 0.3274 & \cellcolor{green!25} 0.3364 & 0.3289 & \cellcolor{green!10} 0.3306 & \cellcolor{green!40} 0.3399 \\
rte & 0.7834 & \cellcolor{green!25} 0.7243 & \cellcolor{green!10} 0.7220 & \cellcolor{green!40} 0.7283 & 0.7157 & 0.7125 \\
winogrande & 0.6772 & 0.6686 & \cellcolor{green!40} 0.6738 & 0.6668 & \cellcolor{green!10} 0.6700 & \cellcolor{green!25} 0.6717 \\
Mean & 0.6574 & 0.6300 & 0.6346 & 0.6288 & 0.6282 & 0.6345 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.55802  |        0.508746 |
| arc_easy      |   0.835859 |        0.781934 |
| boolq         |   0.866055 |        0.843578 |
| hellaswag     |   0.571301 |        0.583344 |
| openbookqa    |   0.31     |        0.339875 |
| rte           |   0.783394 |        0.712545 |
| winogrande    |   0.67719  |        0.671665 |
| Mean          |   0.657403 |        0.634527 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5087 \\
arc_easy & 0.8359 & 0.7819 \\
boolq & 0.8661 & 0.8436 \\
hellaswag & 0.5713 & 0.5833 \\
openbookqa & 0.3100 & 0.3399 \\
rte & 0.7834 & 0.7125 \\
winogrande & 0.6772 & 0.6717 \\
Mean & 0.6574 & 0.6345 \\
\bottomrule
\end{tabular}
```
