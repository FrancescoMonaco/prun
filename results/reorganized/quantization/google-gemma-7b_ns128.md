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

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.441606 |        0.491521 |
| arc_easy      |   0.826178 | 0.72213  |        0.762179 |
| boolq         |   0.836086 | 0.722592 |        0.743119 |
| hellaswag     |   0.606552 | 0.584252 |        0.669121 |
| openbookqa    |   0.32     | 0.346187 |        0.37975  |
| rte           |   0.6787   | 0.538357 |        0.622292 |
| winogrande    |   0.753749 | 0.687352 |        0.717739 |
| Mean          |   0.645651 | 0.577497 |        0.626532 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4416 & \cellcolor{blue!15} \textbf{0.4915} \\
arc_easy & 0.8262 & 0.7221 & \cellcolor{blue!15} \textbf{0.7622} \\
boolq & 0.8361 & 0.7226 & \cellcolor{blue!15} \textbf{0.7431} \\
hellaswag & 0.6066 & 0.5843 & \cellcolor{blue!15} \textbf{0.6691} \\
openbookqa & 0.3200 & 0.3462 & \cellcolor{blue!15} \textbf{0.3798} \\
rte & 0.6787 & 0.5384 & \cellcolor{blue!15} \textbf{0.6223} \\
winogrande & 0.7537 & 0.6874 & \cellcolor{blue!15} \textbf{0.7177} \\
Mean & 0.6457 & 0.5775 & 0.6265 \\
\bottomrule
\end{tabular}
```
