# Results for google/gemma-7b (nsamples=512) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | **0.4887**              | **0.4923**         | 0.4846         | **0.4900** | 0.4877          |
| arc_easy      |     0.8262 | 0.7617                  | **0.7619**         | 0.7543         | **0.7624** | **0.7625**      |
| boolq         |     0.8361 | 0.7499                  | **0.7613**         | **0.7568**     | **0.7567** | 0.7567          |
| hellaswag     |     0.6066 | **0.6658**              | **0.6659**         | 0.6651         | 0.6654     | **0.6732**      |
| openbookqa    |     0.32   | 0.3744                  | **0.3849**         | **0.3754**     | **0.3776** | 0.3742          |
| rte           |     0.6787 | 0.6200                  | **0.6236**         | **0.6403**     | **0.6295** | 0.6151          |
| winogrande    |     0.7537 | 0.7169                  | **0.7194**         | 0.7127         | **0.7212** | **0.7209**      |
| Mean          |     0.6457 | 0.6254                  | 0.6299             | 0.6271         | 0.6290     | 0.6272          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!10} 0.4887 & \cellcolor{green!40} 0.4923 & 0.4846 & \cellcolor{green!25} 0.4900 & 0.4877 \\
arc_easy & 0.8262 & 0.7617 & \cellcolor{green!10} 0.7619 & 0.7543 & \cellcolor{green!25} 0.7624 & \cellcolor{green!40} 0.7625 \\
boolq & 0.8361 & 0.7499 & \cellcolor{green!40} 0.7613 & \cellcolor{green!25} 0.7568 & \cellcolor{green!10} 0.7567 & 0.7567 \\
hellaswag & 0.6066 & \cellcolor{green!10} 0.6658 & \cellcolor{green!25} 0.6659 & 0.6651 & 0.6654 & \cellcolor{green!40} 0.6732 \\
openbookqa & 0.3200 & 0.3744 & \cellcolor{green!40} 0.3849 & \cellcolor{green!10} 0.3754 & \cellcolor{green!25} 0.3776 & 0.3742 \\
rte & 0.6787 & 0.6200 & \cellcolor{green!10} 0.6236 & \cellcolor{green!40} 0.6403 & \cellcolor{green!25} 0.6295 & 0.6151 \\
winogrande & 0.7537 & 0.7169 & \cellcolor{green!10} 0.7194 & 0.7127 & \cellcolor{green!40} 0.7212 & \cellcolor{green!25} 0.7209 \\
Mean & 0.6457 & 0.6254 & 0.6299 & 0.6271 & 0.6290 & 0.6272 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.444246 |        0.487735 |
| arc_easy      |   0.826178 | 0.726786 |        0.762468 |
| boolq         |   0.836086 | 0.725937 |        0.756728 |
| hellaswag     |   0.606552 | 0.588015 |        0.673235 |
| openbookqa    |   0.32     | 0.345062 |        0.37425  |
| rte           |   0.6787   | 0.555505 |        0.615072 |
| winogrande    |   0.753749 | 0.680989 |        0.720896 |
| Mean          |   0.645651 | 0.580934 |        0.627198 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4442 & \cellcolor{blue!15} \textbf{0.4877} \\
arc_easy & 0.8262 & 0.7268 & \cellcolor{blue!15} \textbf{0.7625} \\
boolq & 0.8361 & 0.7259 & \cellcolor{blue!15} \textbf{0.7567} \\
hellaswag & 0.6066 & 0.5880 & \cellcolor{blue!15} \textbf{0.6732} \\
openbookqa & 0.3200 & 0.3451 & \cellcolor{blue!15} \textbf{0.3742} \\
rte & 0.6787 & 0.5555 & \cellcolor{blue!15} \textbf{0.6151} \\
winogrande & 0.7537 & 0.6810 & \cellcolor{blue!15} \textbf{0.7209} \\
Mean & 0.6457 & 0.5809 & 0.6272 \\
\bottomrule
\end{tabular}
```
