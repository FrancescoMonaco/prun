# Results for meta-llama/Llama-3.2-3B (nsamples=16) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | **0.4239**              | **0.4190**         | 0.4171         | 0.3952     | **0.4217**      |
| arc_easy      |     0.7449 | **0.7075**              | **0.7034**         | 0.6996         | 0.6868     | **0.7038**      |
| boolq         |     0.7416 | **0.7143**              | **0.7032**         | **0.7048**     | 0.6745     | 0.6894          |
| hellaswag     |     0.5582 | 0.6234                  | **0.6244**         | **0.6236**     | **0.6238** | 0.6230          |
| openbookqa    |     0.312  | 0.3434                  | **0.3473**         | **0.3479**     | **0.3486** | 0.3438          |
| rte           |     0.5415 | **0.5366**              | **0.5614**         | 0.5203         | 0.5135     | **0.5591**      |
| winogrande    |     0.6938 | **0.6853**              | **0.6858**         | 0.6813         | **0.6831** | 0.6816          |
| Mean          |     0.5742 | 0.5763                  | 0.5778             | 0.5707         | 0.5608     | 0.5746          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!40} 0.4239 & \cellcolor{green!10} 0.4190 & 0.4171 & 0.3952 & \cellcolor{green!25} 0.4217 \\
arc_easy & 0.7449 & \cellcolor{green!40} 0.7075 & \cellcolor{green!10} 0.7034 & 0.6996 & 0.6868 & \cellcolor{green!25} 0.7038 \\
boolq & 0.7416 & \cellcolor{green!40} 0.7143 & \cellcolor{green!10} 0.7032 & \cellcolor{green!25} 0.7048 & 0.6745 & 0.6894 \\
hellaswag & 0.5582 & 0.6234 & \cellcolor{green!40} 0.6244 & \cellcolor{green!10} 0.6236 & \cellcolor{green!25} 0.6238 & 0.6230 \\
openbookqa & 0.3120 & 0.3434 & \cellcolor{green!10} 0.3473 & \cellcolor{green!25} 0.3479 & \cellcolor{green!40} 0.3486 & 0.3438 \\
rte & 0.5415 & \cellcolor{green!10} 0.5366 & \cellcolor{green!40} 0.5614 & 0.5203 & 0.5135 & \cellcolor{green!25} 0.5591 \\
winogrande & 0.6938 & \cellcolor{green!25} 0.6853 & \cellcolor{green!40} 0.6858 & 0.6813 & \cellcolor{green!10} 0.6831 & 0.6816 \\
Mean & 0.5742 & 0.5763 & 0.5778 & 0.5707 & 0.5608 & 0.5746 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.427474 |        0.421662 |
| arc_easy      |   0.744949 |        0.703783 |
| boolq         |   0.74159  |        0.689411 |
| hellaswag     |   0.558156 |        0.622977 |
| openbookqa    |   0.312    |        0.34375  |
| rte           |   0.541516 |        0.559116 |
| winogrande    |   0.693765 |        0.68163  |
| Mean          |   0.574207 |        0.574618 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.4217 \\
arc_easy & 0.7449 & 0.7038 \\
boolq & 0.7416 & 0.6894 \\
hellaswag & 0.5582 & 0.6230 \\
openbookqa & 0.3120 & 0.3438 \\
rte & 0.5415 & 0.5591 \\
winogrande & 0.6938 & 0.6816 \\
Mean & 0.5742 & 0.5746 \\
\bottomrule
\end{tabular}
```
