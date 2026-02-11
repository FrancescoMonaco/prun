# Results for google/gemma-7b (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.3778                  | **0.3893**         | 0.3796         | **0.3840** | **0.4009**      |
| arc_easy      |     0.8262 | 0.6750                  | **0.6855**         | 0.6742         | **0.6823** | **0.7001**      |
| boolq         |     0.8361 | 0.6674                  | **0.6745**         | 0.6735         | **0.6910** | **0.6985**      |
| hellaswag     |     0.6066 | 0.4803                  | **0.5010**         | 0.4827         | **0.5010** | **0.5186**      |
| openbookqa    |     0.32   | **0.3107**              | **0.3130**         | 0.3056         | 0.3095     | **0.3185**      |
| rte           |     0.6787 | 0.4788                  | **0.4937**         | **0.4865**     | 0.4801     | **0.4847**      |
| winogrande    |     0.7537 | 0.6367                  | **0.6461**         | 0.6400         | **0.6547** | **0.6565**      |
| Mean          |     0.6457 | 0.5181                  | 0.5290             | 0.5203         | 0.5289     | 0.5397          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.3778 & \cellcolor{green!25} 0.3893 & 0.3796 & \cellcolor{green!10} 0.3840 & \cellcolor{green!40} 0.4009 \\
arc_easy & 0.8262 & 0.6750 & \cellcolor{green!25} 0.6855 & 0.6742 & \cellcolor{green!10} 0.6823 & \cellcolor{green!40} 0.7001 \\
boolq & 0.8361 & 0.6674 & \cellcolor{green!10} 0.6745 & 0.6735 & \cellcolor{green!25} 0.6910 & \cellcolor{green!40} 0.6985 \\
hellaswag & 0.6066 & 0.4803 & \cellcolor{green!25} 0.5010 & 0.4827 & \cellcolor{green!10} 0.5010 & \cellcolor{green!40} 0.5186 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3107 & \cellcolor{green!25} 0.3130 & 0.3056 & 0.3095 & \cellcolor{green!40} 0.3185 \\
rte & 0.6787 & 0.4788 & \cellcolor{green!40} 0.4937 & \cellcolor{green!25} 0.4865 & 0.4801 & \cellcolor{green!10} 0.4847 \\
winogrande & 0.7537 & 0.6367 & \cellcolor{green!10} 0.6461 & 0.6400 & \cellcolor{green!25} 0.6547 & \cellcolor{green!40} 0.6565 \\
Mean & 0.6457 & 0.5181 & 0.5290 & 0.5203 & 0.5289 & 0.5397 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.441606 |        0.400864 |
| arc_easy      |   0.826178 | 0.72213  |        0.700074 |
| boolq         |   0.836086 | 0.722592 |        0.698547 |
| hellaswag     |   0.606552 | 0.584252 |        0.518578 |
| openbookqa    |   0.32     | 0.346187 |        0.3185   |
| rte           |   0.6787   | 0.538357 |        0.484657 |
| winogrande    |   0.753749 | 0.687352 |        0.656472 |
| Mean          |   0.645651 | 0.577497 |        0.53967  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.4416} & 0.4009 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.7221} & 0.7001 \\
boolq & 0.8361 & \cellcolor{blue!15} \textbf{0.7226} & 0.6985 \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.5843} & 0.5186 \\
openbookqa & 0.3200 & \cellcolor{blue!15} \textbf{0.3462} & 0.3185 \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.5384} & 0.4847 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.6874} & 0.6565 \\
Mean & 0.6457 & 0.5775 & 0.5397 \\
\bottomrule
\end{tabular}
```
