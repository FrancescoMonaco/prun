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

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.48649  |
| arc_easy      |   0.826178 |                0.758733 |
| boolq         |   0.836086 |                0.745413 |
| hellaswag     |   0.606552 |                0.665073 |
| openbookqa    |   0.32     |                0.375    |
| rte           |   0.6787   |                0.615523 |
| winogrande    |   0.753749 |                0.71376  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.4865 \\
arc_easy & 0.8262 & 0.7587 \\
boolq & 0.8361 & 0.7454 \\
hellaswag & 0.6066 & 0.6651 \\
openbookqa & 0.3200 & 0.3750 \\
rte & 0.6787 & 0.6155 \\
winogrande & 0.7537 & 0.7138 \\
\bottomrule
\end{tabular}
```
