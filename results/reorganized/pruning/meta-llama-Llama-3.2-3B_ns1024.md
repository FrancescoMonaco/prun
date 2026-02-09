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

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.427474 |                0.349189 |
| arc_easy      |   0.744949 |                0.625666 |
| boolq         |   0.74159  |                0.676962 |
| hellaswag     |   0.558156 |                0.521418 |
| openbookqa    |   0.312    |                0.302    |
| rte           |   0.541516 |                0.55716  |
| winogrande    |   0.693765 |                0.648908 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4275 & 0.3492 \\
arc_easy & 0.7449 & 0.6257 \\
boolq & 0.7416 & 0.6770 \\
hellaswag & 0.5582 & 0.5214 \\
openbookqa & 0.3120 & 0.3020 \\
rte & 0.5415 & 0.5572 \\
winogrande & 0.6938 & 0.6489 \\
\bottomrule
\end{tabular}
```
