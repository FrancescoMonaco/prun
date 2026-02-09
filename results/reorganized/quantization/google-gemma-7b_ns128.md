# Results for google/gemma-7b (nsamples=128) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | **0.4853**              | 0.4780             | **0.4814**     | 0.4712     | **0.4915**      |
| arc_easy      |     0.8262 | **0.7584**              | **0.7502**         | 0.7501         | 0.7303     | **0.7622**      |
| boolq         |     0.8361 | **0.7539**              | **0.7636**         | 0.7234         | **0.7462** | 0.7431          |
| hellaswag     |     0.6066 | **0.6587**              | 0.6533             | 0.6508         | **0.6561** | **0.6691**      |
| openbookqa    |     0.32   | 0.3660                  | **0.3695**         | 0.3678         | **0.3688** | **0.3798**      |
| rte           |     0.6787 | **0.6313**              | **0.6241**         | 0.6151         | 0.5984     | **0.6223**      |
| winogrande    |     0.7537 | **0.7230**              | 0.7171             | 0.7148         | **0.7176** | **0.7177**      |
| Mean          |     0.6457 | 0.6252                  | 0.6223             | 0.6148         | 0.6126     | 0.6265          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!25} 0.4853 & 0.4780 & \cellcolor{green!10} 0.4814 & 0.4712 & \cellcolor{green!40} 0.4915 \\
arc_easy & 0.8262 & \cellcolor{green!25} 0.7584 & \cellcolor{green!10} 0.7502 & 0.7501 & 0.7303 & \cellcolor{green!40} 0.7622 \\
boolq & 0.8361 & \cellcolor{green!25} 0.7539 & \cellcolor{green!40} 0.7636 & 0.7234 & \cellcolor{green!10} 0.7462 & 0.7431 \\
hellaswag & 0.6066 & \cellcolor{green!25} 0.6587 & 0.6533 & 0.6508 & \cellcolor{green!10} 0.6561 & \cellcolor{green!40} 0.6691 \\
openbookqa & 0.3200 & 0.3660 & \cellcolor{green!25} 0.3695 & 0.3678 & \cellcolor{green!10} 0.3688 & \cellcolor{green!40} 0.3798 \\
rte & 0.6787 & \cellcolor{green!40} 0.6313 & \cellcolor{green!25} 0.6241 & 0.6151 & 0.5984 & \cellcolor{green!10} 0.6223 \\
winogrande & 0.7537 & \cellcolor{green!40} 0.7230 & 0.7171 & 0.7148 & \cellcolor{green!10} 0.7176 & \cellcolor{green!25} 0.7177 \\
Mean & 0.6457 & 0.6252 & 0.6223 & 0.6148 & 0.6126 & 0.6265 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.48521  |
| arc_easy      |   0.826178 |                0.756559 |
| boolq         |   0.836086 |                0.754128 |
| hellaswag     |   0.606552 |                0.657754 |
| openbookqa    |   0.32     |                0.365833 |
| rte           |   0.6787   |                0.631167 |
| winogrande    |   0.753749 |                0.722047 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.4852 \\
arc_easy & 0.8262 & 0.7566 \\
boolq & 0.8361 & 0.7541 \\
hellaswag & 0.6066 & 0.6578 \\
openbookqa & 0.3200 & 0.3658 \\
rte & 0.6787 & 0.6312 \\
winogrande & 0.7537 & 0.7220 \\
\bottomrule
\end{tabular}
```
