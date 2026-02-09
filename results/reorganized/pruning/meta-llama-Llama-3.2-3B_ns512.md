# Results for meta-llama/Llama-3.2-3B (nsamples=512) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | 0.3501                  | **0.3567**         | 0.3491         | **0.3515** | **0.3554**      |
| arc_easy      |     0.7449 | 0.6229                  | **0.6296**         | 0.6235         | **0.6274** | **0.6364**      |
| boolq         |     0.7416 | **0.6808**              | 0.6677             | 0.6670         | **0.6731** | **0.6783**      |
| hellaswag     |     0.5582 | **0.5214**              | **0.5261**         | 0.5207         | 0.5210     | **0.5301**      |
| openbookqa    |     0.312  | **0.3041**              | **0.3099**         | 0.3039         | 0.3037     | **0.3096**      |
| rte           |     0.5415 | 0.5451                  | **0.5686**         | **0.5523**     | 0.5460     | **0.5560**      |
| winogrande    |     0.6938 | **0.6503**              | 0.6478             | **0.6507**     | 0.6484     | **0.6554**      |
| Mean          |     0.5742 | 0.5250                  | 0.5295             | 0.5239         | 0.5244     | 0.5316          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.3501 & \cellcolor{green!40} 0.3567 & 0.3491 & \cellcolor{green!10} 0.3515 & \cellcolor{green!25} 0.3554 \\
arc_easy & 0.7449 & 0.6229 & \cellcolor{green!25} 0.6296 & 0.6235 & \cellcolor{green!10} 0.6274 & \cellcolor{green!40} 0.6364 \\
boolq & 0.7416 & \cellcolor{green!40} 0.6808 & 0.6677 & 0.6670 & \cellcolor{green!10} 0.6731 & \cellcolor{green!25} 0.6783 \\
hellaswag & 0.5582 & \cellcolor{green!10} 0.5214 & \cellcolor{green!25} 0.5261 & 0.5207 & 0.5210 & \cellcolor{green!40} 0.5301 \\
openbookqa & 0.3120 & \cellcolor{green!10} 0.3041 & \cellcolor{green!40} 0.3099 & 0.3039 & 0.3037 & \cellcolor{green!25} 0.3096 \\
rte & 0.5415 & 0.5451 & \cellcolor{green!40} 0.5686 & \cellcolor{green!10} 0.5523 & 0.5460 & \cellcolor{green!25} 0.5560 \\
winogrande & 0.6938 & \cellcolor{green!10} 0.6503 & 0.6478 & \cellcolor{green!25} 0.6507 & 0.6484 & \cellcolor{green!40} 0.6554 \\
Mean & 0.5742 & 0.5250 & 0.5295 & 0.5239 & 0.5244 & 0.5316 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.427474 |                0.350114 |
| arc_easy      |   0.744949 |                0.621843 |
| boolq         |   0.74159  |                0.68104  |
| hellaswag     |   0.558156 |                0.521476 |
| openbookqa    |   0.312    |                0.303833 |
| rte           |   0.541516 |                0.548736 |
| winogrande    |   0.693765 |                0.650487 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4275 & 0.3501 \\
arc_easy & 0.7449 & 0.6218 \\
boolq & 0.7416 & 0.6810 \\
hellaswag & 0.5582 & 0.5215 \\
openbookqa & 0.3120 & 0.3038 \\
rte & 0.5415 & 0.5487 \\
winogrande & 0.6938 & 0.6505 \\
\bottomrule
\end{tabular}
```
