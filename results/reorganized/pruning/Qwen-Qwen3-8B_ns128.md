# Results for Qwen/Qwen3-8B (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5026                  | **0.5078**         | **0.5061**     | 0.4939     | **0.5134**      |
| arc_easy      |     0.8359 | **0.7775**              | **0.7822**         | 0.7756         | 0.7705     | **0.7825**      |
| boolq         |     0.8661 | **0.8432**              | **0.8441**         | 0.8413         | **0.8443** | 0.8422          |
| hellaswag     |     0.5713 | 0.5763                  | **0.5812**         | 0.5761         | **0.5783** | **0.5818**      |
| openbookqa    |     0.31   | 0.3289                  | **0.3405**         | **0.3369**     | 0.3292     | **0.3386**      |
| rte           |     0.7834 | 0.7062                  | 0.7103             | **0.7229**     | **0.7211** | **0.7301**      |
| winogrande    |     0.6772 | 0.6709                  | **0.6787**         | **0.6732**     | 0.6701     | **0.6749**      |
| Mean          |     0.6574 | 0.6294                  | 0.6350             | 0.6332         | 0.6296     | 0.6377          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5026 & \cellcolor{green!25} 0.5078 & \cellcolor{green!10} 0.5061 & 0.4939 & \cellcolor{green!40} 0.5134 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.7775 & \cellcolor{green!25} 0.7822 & 0.7756 & 0.7705 & \cellcolor{green!40} 0.7825 \\
boolq & 0.8661 & \cellcolor{green!10} 0.8432 & \cellcolor{green!25} 0.8441 & 0.8413 & \cellcolor{green!40} 0.8443 & 0.8422 \\
hellaswag & 0.5713 & 0.5763 & \cellcolor{green!25} 0.5812 & 0.5761 & \cellcolor{green!10} 0.5783 & \cellcolor{green!40} 0.5818 \\
openbookqa & 0.3100 & 0.3289 & \cellcolor{green!40} 0.3405 & \cellcolor{green!10} 0.3369 & 0.3292 & \cellcolor{green!25} 0.3386 \\
rte & 0.7834 & 0.7062 & 0.7103 & \cellcolor{green!25} 0.7229 & \cellcolor{green!10} 0.7211 & \cellcolor{green!40} 0.7301 \\
winogrande & 0.6772 & 0.6709 & \cellcolor{green!40} 0.6787 & \cellcolor{green!10} 0.6732 & 0.6701 & \cellcolor{green!25} 0.6749 \\
Mean & 0.6574 & 0.6294 & 0.6350 & 0.6332 & 0.6296 & 0.6377 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.55802  | 0.520158 |        0.513439 |
| arc_easy      |   0.835859 | 0.78805  |        0.782539 |
| boolq         |   0.866055 | 0.848567 |        0.84224  |
| hellaswag     |   0.571301 | 0.609157 |        0.581825 |
| openbookqa    |   0.31     | 0.346375 |        0.338625 |
| rte           |   0.783394 | 0.738267 |        0.730144 |
| winogrande    |   0.67719  | 0.673441 |        0.674921 |
| Mean          |   0.657403 | 0.646288 |        0.637676 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.5580 & \cellcolor{blue!15} \textbf{0.5202} & 0.5134 \\
arc_easy & 0.8359 & \cellcolor{blue!15} \textbf{0.7880} & 0.7825 \\
boolq & 0.8661 & \cellcolor{blue!15} \textbf{0.8486} & 0.8422 \\
hellaswag & 0.5713 & \cellcolor{blue!15} \textbf{0.6092} & 0.5818 \\
openbookqa & 0.3100 & \cellcolor{blue!15} \textbf{0.3464} & 0.3386 \\
rte & 0.7834 & \cellcolor{blue!15} \textbf{0.7383} & 0.7301 \\
winogrande & 0.6772 & 0.6734 & \cellcolor{blue!15} \textbf{0.6749} \\
Mean & 0.6574 & 0.6463 & 0.6377 \\
\bottomrule
\end{tabular}
```
