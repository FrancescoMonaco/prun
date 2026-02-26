# Results for Qwen/Qwen3-8B (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | **0.5203**              | **0.5226**         | 0.5028         | 0.5137     | **0.5147**      |
| arc_easy      |     0.8359 | **0.7856**              | **0.7870**         | 0.7775         | 0.7845     | **0.7863**      |
| boolq         |     0.8661 | 0.8445                  | **0.8466**         | 0.8414         | **0.8489** | **0.8480**      |
| hellaswag     |     0.5713 | 0.5823                  | **0.5870**         | 0.5785         | **0.5832** | **0.5906**      |
| openbookqa    |     0.31   | 0.3295                  | **0.3365**         | 0.3280         | **0.3340** | **0.3415**      |
| rte           |     0.7834 | 0.7148                  | **0.7184**         | **0.7202**     | **0.7184** | 0.7112          |
| winogrande    |     0.6772 | **0.6772**              | 0.6736             | **0.6756**     | 0.6736     | **0.6752**      |
| Mean          |     0.6574 | 0.6363                  | 0.6388             | 0.6320         | 0.6366     | 0.6382          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!25} 0.5203 & \cellcolor{green!40} 0.5226 & 0.5028 & 0.5137 & \cellcolor{green!10} 0.5147 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.7856 & \cellcolor{green!40} 0.7870 & 0.7775 & 0.7845 & \cellcolor{green!25} 0.7863 \\
boolq & 0.8661 & 0.8445 & \cellcolor{green!10} 0.8466 & 0.8414 & \cellcolor{green!40} 0.8489 & \cellcolor{green!25} 0.8480 \\
hellaswag & 0.5713 & 0.5823 & \cellcolor{green!25} 0.5870 & 0.5785 & \cellcolor{green!10} 0.5832 & \cellcolor{green!40} 0.5906 \\
openbookqa & 0.3100 & 0.3295 & \cellcolor{green!25} 0.3365 & 0.3280 & \cellcolor{green!10} 0.3340 & \cellcolor{green!40} 0.3415 \\
rte & 0.7834 & 0.7148 & \cellcolor{green!25} 0.7184 & \cellcolor{green!40} 0.7202 & \cellcolor{green!10} 0.7184 & 0.7112 \\
winogrande & 0.6772 & \cellcolor{green!40} 0.6772 & 0.6736 & \cellcolor{green!25} 0.6756 & 0.6736 & \cellcolor{green!10} 0.6752 \\
Mean & 0.6574 & 0.6363 & 0.6388 & 0.6320 & 0.6366 & 0.6382 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.55802  | 0.552474 |        0.514718 |
| arc_easy      |   0.835859 | 0.819602 |        0.786301 |
| boolq         |   0.866055 | 0.862462 |        0.848012 |
| hellaswag     |   0.571301 | 0.649746 |        0.59057  |
| openbookqa    |   0.31     | 0.36525  |        0.3415   |
| rte           |   0.783394 | 0.767148 |        0.711191 |
| winogrande    |   0.67719  | 0.677979 |        0.675217 |
| Mean          |   0.657403 | 0.670666 |        0.638216 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.5580 & \cellcolor{blue!15} \textbf{0.5525} & 0.5147 \\
arc_easy & 0.8359 & \cellcolor{blue!15} \textbf{0.8196} & 0.7863 \\
boolq & 0.8661 & \cellcolor{blue!15} \textbf{0.8625} & 0.8480 \\
hellaswag & 0.5713 & \cellcolor{blue!15} \textbf{0.6497} & 0.5906 \\
openbookqa & 0.3100 & \cellcolor{blue!15} \textbf{0.3652} & 0.3415 \\
rte & 0.7834 & \cellcolor{blue!15} \textbf{0.7671} & 0.7112 \\
winogrande & 0.6772 & \cellcolor{blue!15} \textbf{0.6780} & 0.6752 \\
Mean & 0.6574 & 0.6707 & 0.6382 \\
\bottomrule
\end{tabular}
```
