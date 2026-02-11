# Results for meta-llama/Llama-3.2-3B (nsamples=1024) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | **0.3504**              | **0.3527**         | 0.3497         | 0.3502     | **0.3534**      |
| arc_easy      |     0.7449 | 0.6265                  | **0.6292**         | 0.6240         | **0.6268** | **0.6292**      |
| boolq         |     0.7416 | **0.6775**              | 0.6641             | 0.6610         | **0.6756** | **0.6785**      |
| hellaswag     |     0.5582 | 0.5212                  | **0.5246**         | 0.5210         | **0.5213** | **0.5261**      |
| openbookqa    |     0.312  | 0.3019                  | **0.3035**         | **0.3034**     | **0.3050** | 0.3033          |
| rte           |     0.5415 | **0.5532**              | **0.5609**         | 0.5474         | 0.5465     | **0.5532**      |
| winogrande    |     0.6938 | 0.6470                  | **0.6484**         | **0.6494**     | 0.6483     | **0.6525**      |
| Mean          |     0.5742 | 0.5254                  | 0.5262             | 0.5223         | 0.5248     | 0.5280          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!10} 0.3504 & \cellcolor{green!25} 0.3527 & 0.3497 & 0.3502 & \cellcolor{green!40} 0.3534 \\
arc_easy & 0.7449 & 0.6265 & \cellcolor{green!25} 0.6292 & 0.6240 & \cellcolor{green!10} 0.6268 & \cellcolor{green!40} 0.6292 \\
boolq & 0.7416 & \cellcolor{green!25} 0.6775 & 0.6641 & 0.6610 & \cellcolor{green!10} 0.6756 & \cellcolor{green!40} 0.6785 \\
hellaswag & 0.5582 & 0.5212 & \cellcolor{green!25} 0.5246 & 0.5210 & \cellcolor{green!10} 0.5213 & \cellcolor{green!40} 0.5261 \\
openbookqa & 0.3120 & 0.3019 & \cellcolor{green!25} 0.3035 & \cellcolor{green!10} 0.3034 & \cellcolor{green!40} 0.3050 & 0.3033 \\
rte & 0.5415 & \cellcolor{green!25} 0.5532 & \cellcolor{green!40} 0.5609 & 0.5474 & 0.5465 & \cellcolor{green!10} 0.5532 \\
winogrande & 0.6938 & 0.6470 & \cellcolor{green!10} 0.6484 & \cellcolor{green!25} 0.6494 & 0.6483 & \cellcolor{green!40} 0.6525 \\
Mean & 0.5742 & 0.5254 & 0.5262 & 0.5223 & 0.5248 & 0.5280 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.427474 | 0.381853 |        0.353402 |
| arc_easy      |   0.744949 | 0.662234 |        0.629235 |
| boolq         |   0.74159  | 0.693502 |        0.678479 |
| hellaswag     |   0.558156 | 0.563175 |        0.526115 |
| openbookqa    |   0.312    | 0.3195   |        0.30325  |
| rte           |   0.541516 | 0.548511 |        0.553249 |
| winogrande    |   0.693765 | 0.661109 |        0.652526 |
| Mean          |   0.574207 | 0.547126 |        0.528037 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{blue!15} \textbf{0.3819} & 0.3534 \\
arc_easy & 0.7449 & \cellcolor{blue!15} \textbf{0.6622} & 0.6292 \\
boolq & 0.7416 & \cellcolor{blue!15} \textbf{0.6935} & 0.6785 \\
hellaswag & 0.5582 & \cellcolor{blue!15} \textbf{0.5632} & 0.5261 \\
openbookqa & 0.3120 & \cellcolor{blue!15} \textbf{0.3195} & 0.3033 \\
rte & 0.5415 & 0.5485 & \cellcolor{blue!15} \textbf{0.5532} \\
winogrande & 0.6938 & \cellcolor{blue!15} \textbf{0.6611} & 0.6525 \\
Mean & 0.5742 & 0.5471 & 0.5280 \\
\bottomrule
\end{tabular}
```
