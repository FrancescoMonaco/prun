# Results for google/gemma-7b (nsamples=1024) - pruning

## Average across Calibration Groups

| task          |   original | dictionary   | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:-------------|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.2544       | 0.3803                  | **0.3830**         | 0.3779         | **0.3804** | **0.3884**      |
| arc_easy      |     0.8262 | 0.4920       | **0.6782**              | 0.6779             | **0.6791**     | 0.6766     | **0.6866**      |
| boolq         |     0.8361 | 0.6363       | **0.6789**              | 0.6726             | **0.6729**     | 0.6693     | **0.6853**      |
| hellaswag     |     0.6066 | 0.3278       | 0.4834                  | **0.4876**         | 0.4833         | **0.4837** | **0.4976**      |
| openbookqa    |     0.32   | 0.2540       | 0.3107                  | **0.3145**         | **0.3116**     | 0.3101     | **0.3163**      |
| rte           |     0.6787 | **0.5084**   | **0.4860**              | **0.4901**         | 0.4819         | 0.4801     | 0.4810          |
| winogrande    |     0.7537 | 0.5446       | **0.6381**              | **0.6388**         | 0.6371         | 0.6365     | **0.6414**      |
| Mean          |     0.6457 | 0.4311       | 0.5222                  | 0.5235             | 0.5206         | 0.5195     | 0.5281          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & dictionary & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.2544 & 0.3803 & \cellcolor{green!25} 0.3830 & 0.3779 & \cellcolor{green!10} 0.3804 & \cellcolor{green!40} 0.3884 \\
arc_easy & 0.8262 & 0.4920 & \cellcolor{green!10} 0.6782 & 0.6779 & \cellcolor{green!25} 0.6791 & 0.6766 & \cellcolor{green!40} 0.6866 \\
boolq & 0.8361 & 0.6363 & \cellcolor{green!25} 0.6789 & 0.6726 & \cellcolor{green!10} 0.6729 & 0.6693 & \cellcolor{green!40} 0.6853 \\
hellaswag & 0.6066 & 0.3278 & 0.4834 & \cellcolor{green!25} 0.4876 & 0.4833 & \cellcolor{green!10} 0.4837 & \cellcolor{green!40} 0.4976 \\
openbookqa & 0.3200 & 0.2540 & 0.3107 & \cellcolor{green!25} 0.3145 & \cellcolor{green!10} 0.3116 & 0.3101 & \cellcolor{green!40} 0.3163 \\
rte & 0.6787 & \cellcolor{green!40} 0.5084 & \cellcolor{green!10} 0.4860 & \cellcolor{green!25} 0.4901 & 0.4819 & 0.4801 & 0.4810 \\
winogrande & 0.7537 & 0.5446 & \cellcolor{green!10} 0.6381 & \cellcolor{green!25} 0.6388 & 0.6371 & 0.6365 & \cellcolor{green!40} 0.6414 \\
Mean & 0.6457 & 0.4311 & 0.5222 & 0.5235 & 0.5206 & 0.5195 & 0.5281 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.440833 |        0.388439 |
| arc_easy      |   0.826178 | 0.723143 |        0.686606 |
| boolq         |   0.836086 | 0.721235 |        0.685321 |
| hellaswag     |   0.606552 | 0.582152 |        0.497598 |
| openbookqa    |   0.32     | 0.3395   |        0.31625  |
| rte           |   0.6787   | 0.550993 |        0.481047 |
| winogrande    |   0.753749 | 0.680841 |        0.641377 |
| Mean          |   0.645651 | 0.576957 |        0.528091 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.4408} & 0.3884 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.7231} & 0.6866 \\
boolq & 0.8361 & \cellcolor{blue!15} \textbf{0.7212} & 0.6853 \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.5822} & 0.4976 \\
openbookqa & 0.3200 & \cellcolor{blue!15} \textbf{0.3395} & 0.3163 \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.5510} & 0.4810 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.6808} & 0.6414 \\
Mean & 0.6457 & 0.5770 & 0.5281 \\
\bottomrule
\end{tabular}
```
