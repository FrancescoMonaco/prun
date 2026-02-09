# Results for Qwen/Qwen3-8B (nsamples=512) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.4990                  | **0.5093**         | 0.5045         | **0.5071** | **0.5093**      |
| arc_easy      |     0.8359 | **0.7778**              | **0.7846**         | 0.7761         | 0.7777     | **0.7816**      |
| boolq         |     0.8661 | **0.8453**              | **0.8454**         | **0.8439**     | 0.8437     | 0.8433          |
| hellaswag     |     0.5713 | 0.5772                  | **0.5826**         | 0.5763         | **0.5776** | **0.5836**      |
| openbookqa    |     0.31   | **0.3340**              | **0.3347**         | 0.3306         | 0.3327     | **0.3356**      |
| rte           |     0.7834 | 0.7125                  | 0.7171             | **0.7234**     | **0.7184** | **0.7207**      |
| winogrande    |     0.6772 | **0.6736**              | **0.6765**         | 0.6711         | **0.6742** | 0.6727          |
| Mean          |     0.6574 | 0.6314                  | 0.6357             | 0.6323         | 0.6331     | 0.6352          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.4990 & \cellcolor{green!40} 0.5093 & 0.5045 & \cellcolor{green!10} 0.5071 & \cellcolor{green!25} 0.5093 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.7778 & \cellcolor{green!40} 0.7846 & 0.7761 & 0.7777 & \cellcolor{green!25} 0.7816 \\
boolq & 0.8661 & \cellcolor{green!25} 0.8453 & \cellcolor{green!40} 0.8454 & \cellcolor{green!10} 0.8439 & 0.8437 & 0.8433 \\
hellaswag & 0.5713 & 0.5772 & \cellcolor{green!25} 0.5826 & 0.5763 & \cellcolor{green!10} 0.5776 & \cellcolor{green!40} 0.5836 \\
openbookqa & 0.3100 & \cellcolor{green!10} 0.3340 & \cellcolor{green!25} 0.3347 & 0.3306 & 0.3327 & \cellcolor{green!40} 0.3356 \\
rte & 0.7834 & 0.7125 & 0.7171 & \cellcolor{green!40} 0.7234 & \cellcolor{green!10} 0.7184 & \cellcolor{green!25} 0.7207 \\
winogrande & 0.6772 & \cellcolor{green!10} 0.6736 & \cellcolor{green!40} 0.6765 & 0.6711 & \cellcolor{green!25} 0.6742 & 0.6727 \\
Mean & 0.6574 & 0.6314 & 0.6357 & 0.6323 & 0.6331 & 0.6352 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.55802  |                0.496658 |
| arc_easy      |   0.835859 |                0.778269 |
| boolq         |   0.866055 |                0.845617 |
| hellaswag     |   0.571301 |                0.577674 |
| openbookqa    |   0.31     |                0.333333 |
| rte           |   0.783394 |                0.712395 |
| winogrande    |   0.67719  |                0.672586 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.5580 & 0.4967 \\
arc_easy & 0.8359 & 0.7783 \\
boolq & 0.8661 & 0.8456 \\
hellaswag & 0.5713 & 0.5777 \\
openbookqa & 0.3100 & 0.3333 \\
rte & 0.7834 & 0.7124 \\
winogrande & 0.6772 & 0.6726 \\
\bottomrule
\end{tabular}
```
