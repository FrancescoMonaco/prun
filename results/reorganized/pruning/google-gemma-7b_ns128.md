# Results for google/gemma-7b (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | **0.3901**              | **0.4117**         | 0.3803         | 0.3878     | **0.4123**      |
| arc_easy      |     0.8262 | 0.6956                  | **0.7111**         | 0.6810         | **0.6958** | **0.7163**      |
| boolq         |     0.8361 | 0.6852                  | **0.7028**         | 0.6797         | **0.6972** | **0.7003**      |
| hellaswag     |     0.6066 | 0.5057                  | **0.5376**         | 0.4919         | **0.5092** | **0.5364**      |
| openbookqa    |     0.32   | **0.3210**              | 0.3165             | 0.3120         | **0.3280** | **0.3200**      |
| rte           |     0.6787 | **0.4892**              | **0.4819**         | **0.4964**     | 0.4711     | 0.4639          |
| winogrande    |     0.7537 | 0.6527                  | **0.6622**         | 0.6464         | **0.6555** | **0.6669**      |
| Mean          |     0.6457 | 0.5342                  | 0.5463             | 0.5268         | 0.5350     | 0.5452          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!10} 0.3901 & \cellcolor{green!25} 0.4117 & 0.3803 & 0.3878 & \cellcolor{green!40} 0.4123 \\
arc_easy & 0.8262 & 0.6956 & \cellcolor{green!25} 0.7111 & 0.6810 & \cellcolor{green!10} 0.6958 & \cellcolor{green!40} 0.7163 \\
boolq & 0.8361 & 0.6852 & \cellcolor{green!40} 0.7028 & 0.6797 & \cellcolor{green!10} 0.6972 & \cellcolor{green!25} 0.7003 \\
hellaswag & 0.6066 & 0.5057 & \cellcolor{green!40} 0.5376 & 0.4919 & \cellcolor{green!10} 0.5092 & \cellcolor{green!25} 0.5364 \\
openbookqa & 0.3200 & \cellcolor{green!25} 0.3210 & 0.3165 & 0.3120 & \cellcolor{green!40} 0.3280 & \cellcolor{green!10} 0.3200 \\
rte & 0.6787 & \cellcolor{green!25} 0.4892 & \cellcolor{green!10} 0.4819 & \cellcolor{green!40} 0.4964 & 0.4711 & 0.4639 \\
winogrande & 0.7537 & 0.6527 & \cellcolor{green!25} 0.6622 & 0.6464 & \cellcolor{green!10} 0.6555 & \cellcolor{green!40} 0.6669 \\
Mean & 0.6457 & 0.5342 & 0.5463 & 0.5268 & 0.5350 & 0.5452 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.516532 |        0.412329 |
| arc_easy      |   0.826178 | 0.807134 |        0.71633  |
| boolq         |   0.836086 | 0.812997 |        0.700306 |
| hellaswag     |   0.606552 | 0.698616 |        0.536372 |
| openbookqa    |   0.32     | 0.38475  |        0.32     |
| rte           |   0.6787   | 0.67509  |        0.463899 |
| winogrande    |   0.753749 | 0.742699 |        0.66693  |
| Mean          |   0.645651 | 0.662545 |        0.545167 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.5165} & 0.4123 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.8071} & 0.7163 \\
boolq & 0.8361 & \cellcolor{blue!15} \textbf{0.8130} & 0.7003 \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.6986} & 0.5364 \\
openbookqa & 0.3200 & \cellcolor{blue!15} \textbf{0.3848} & 0.3200 \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.6751} & 0.4639 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.7427} & 0.6669 \\
Mean & 0.6457 & 0.6625 & 0.5452 \\
\bottomrule
\end{tabular}
```
