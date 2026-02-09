# Results for google/gemma-7b (nsamples=1024) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.3803                  | **0.3830**         | 0.3779         | **0.3804** | **0.3884**      |
| arc_easy      |     0.8262 | **0.6782**              | 0.6779             | **0.6791**     | 0.6766     | **0.6866**      |
| boolq         |     0.8361 | **0.6789**              | 0.6726             | **0.6729**     | 0.6693     | **0.6853**      |
| hellaswag     |     0.6066 | 0.4834                  | **0.4876**         | 0.4833         | **0.4837** | **0.4976**      |
| openbookqa    |     0.32   | 0.3107                  | **0.3145**         | **0.3116**     | 0.3101     | **0.3163**      |
| rte           |     0.6787 | **0.4860**              | **0.4901**         | **0.4819**     | 0.4801     | 0.4810          |
| winogrande    |     0.7537 | **0.6381**              | **0.6388**         | 0.6371         | 0.6365     | **0.6414**      |
| Mean          |     0.6457 | 0.5222                  | 0.5235             | 0.5206         | 0.5195     | 0.5281          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.3803 & \cellcolor{green!25} 0.3830 & 0.3779 & \cellcolor{green!10} 0.3804 & \cellcolor{green!40} 0.3884 \\
arc_easy & 0.8262 & \cellcolor{green!10} 0.6782 & 0.6779 & \cellcolor{green!25} 0.6791 & 0.6766 & \cellcolor{green!40} 0.6866 \\
boolq & 0.8361 & \cellcolor{green!25} 0.6789 & 0.6726 & \cellcolor{green!10} 0.6729 & 0.6693 & \cellcolor{green!40} 0.6853 \\
hellaswag & 0.6066 & 0.4834 & \cellcolor{green!25} 0.4876 & 0.4833 & \cellcolor{green!10} 0.4837 & \cellcolor{green!40} 0.4976 \\
openbookqa & 0.3200 & 0.3107 & \cellcolor{green!25} 0.3145 & \cellcolor{green!10} 0.3116 & 0.3101 & \cellcolor{green!40} 0.3163 \\
rte & 0.6787 & \cellcolor{green!25} 0.4860 & \cellcolor{green!40} 0.4901 & \cellcolor{green!10} 0.4819 & 0.4801 & 0.4810 \\
winogrande & 0.7537 & \cellcolor{green!10} 0.6381 & \cellcolor{green!25} 0.6388 & 0.6371 & 0.6365 & \cellcolor{green!40} 0.6414 \\
Mean & 0.6457 & 0.5222 & 0.5235 & 0.5206 & 0.5195 & 0.5281 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.379906 |
| arc_easy      |   0.826178 |                0.676592 |
| boolq         |   0.836086 |                0.677523 |
| hellaswag     |   0.606552 |                0.484872 |
| openbookqa    |   0.32     |                0.308667 |
| rte           |   0.6787   |                0.484958 |
| winogrande    |   0.753749 |                0.637332 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.3799 \\
arc_easy & 0.8262 & 0.6766 \\
boolq & 0.8361 & 0.6775 \\
hellaswag & 0.6066 & 0.4849 \\
openbookqa & 0.3200 & 0.3087 \\
rte & 0.6787 & 0.4850 \\
winogrande & 0.7537 & 0.6373 \\
\bottomrule
\end{tabular}
```
