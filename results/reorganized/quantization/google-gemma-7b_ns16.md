# Results for google/gemma-7b (nsamples=16) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | **0.4668**              | 0.4595             | 0.4564         | **0.4749** | **0.4675**      |
| arc_easy      |     0.8262 | **0.7216**              | 0.7075             | 0.6993         | **0.7270** | **0.7255**      |
| boolq         |     0.8361 | 0.7112                  | **0.7356**         | **0.7213**     | **0.7269** | 0.7119          |
| hellaswag     |     0.6066 | **0.6311**              | 0.6218             | 0.6063         | **0.6296** | **0.6441**      |
| openbookqa    |     0.32   | **0.3680**              | 0.3594             | **0.3705**     | 0.3655     | **0.3674**      |
| rte           |     0.6787 | 0.5736                  | 0.6047             | **0.6051**     | **0.6214** | **0.6097**      |
| winogrande    |     0.7537 | **0.7046**              | 0.7030             | 0.6912         | **0.7053** | **0.7095**      |
| Mean          |     0.6457 | 0.5967                  | 0.5988             | 0.5929         | 0.6072     | 0.6051          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!10} 0.4668 & 0.4595 & 0.4564 & \cellcolor{green!40} 0.4749 & \cellcolor{green!25} 0.4675 \\
arc_easy & 0.8262 & \cellcolor{green!10} 0.7216 & 0.7075 & 0.6993 & \cellcolor{green!40} 0.7270 & \cellcolor{green!25} 0.7255 \\
boolq & 0.8361 & 0.7112 & \cellcolor{green!40} 0.7356 & \cellcolor{green!10} 0.7213 & \cellcolor{green!25} 0.7269 & 0.7119 \\
hellaswag & 0.6066 & \cellcolor{green!25} 0.6311 & 0.6218 & 0.6063 & \cellcolor{green!10} 0.6296 & \cellcolor{green!40} 0.6441 \\
openbookqa & 0.3200 & \cellcolor{green!25} 0.3680 & 0.3594 & \cellcolor{green!40} 0.3705 & 0.3655 & \cellcolor{green!10} 0.3674 \\
rte & 0.6787 & 0.5736 & 0.6047 & \cellcolor{green!10} 0.6051 & \cellcolor{green!40} 0.6214 & \cellcolor{green!25} 0.6097 \\
winogrande & 0.7537 & \cellcolor{green!10} 0.7046 & 0.7030 & 0.6912 & \cellcolor{green!25} 0.7053 & \cellcolor{green!40} 0.7095 \\
Mean & 0.6457 & 0.5967 & 0.5988 & 0.5929 & 0.6072 & 0.6051 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.498294 |        0.467523 |
| arc_easy      |   0.826178 |        0.725458 |
| boolq         |   0.836086 |        0.71185  |
| hellaswag     |   0.606552 |        0.64412  |
| openbookqa    |   0.32     |        0.367375 |
| rte           |   0.6787   |        0.609657 |
| winogrande    |   0.753749 |        0.709451 |
| Mean          |   0.645651 |        0.605062 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4675 \\
arc_easy & 0.8262 & 0.7255 \\
boolq & 0.8361 & 0.7119 \\
hellaswag & 0.6066 & 0.6441 \\
openbookqa & 0.3200 & 0.3674 \\
rte & 0.6787 & 0.6097 \\
winogrande & 0.7537 & 0.7095 \\
Mean & 0.6457 & 0.6051 \\
\bottomrule
\end{tabular}
```
