# Results for meta-llama/Llama-3.2-3B (nsamples=1024) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | **0.4306**              | 0.4262             | **0.4289**     | 0.4280     | **0.4283**      |
| arc_easy      |     0.7449 | **0.7152**              | 0.7081             | 0.7113         | **0.7188** | **0.7153**      |
| boolq         |     0.7416 | **0.7252**              | **0.7324**         | 0.7106         | 0.7170     | **0.7295**      |
| hellaswag     |     0.5582 | **0.6345**              | 0.6342             | 0.6331         | **0.6345** | **0.6350**      |
| openbookqa    |     0.312  | **0.3543**              | 0.3489             | 0.3514         | **0.3549** | **0.3550**      |
| rte           |     0.5415 | 0.5221                  | 0.5176             | **0.5469**     | **0.5560** | **0.5266**      |
| winogrande    |     0.6938 | 0.6862                  | **0.6873**         | **0.6893**     | 0.6871     | **0.6947**      |
| Mean          |     0.5742 | 0.5811                  | 0.5792             | 0.5817         | 0.5852     | 0.5835          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & \cellcolor{green!40} 0.4306 & 0.4262 & \cellcolor{green!25} 0.4289 & 0.4280 & \cellcolor{green!10} 0.4283 \\
arc_easy & 0.7449 & \cellcolor{green!10} 0.7152 & 0.7081 & 0.7113 & \cellcolor{green!40} 0.7188 & \cellcolor{green!25} 0.7153 \\
boolq & 0.7416 & \cellcolor{green!10} 0.7252 & \cellcolor{green!40} 0.7324 & 0.7106 & 0.7170 & \cellcolor{green!25} 0.7295 \\
hellaswag & 0.5582 & \cellcolor{green!10} 0.6345 & 0.6342 & 0.6331 & \cellcolor{green!25} 0.6345 & \cellcolor{green!40} 0.6350 \\
openbookqa & 0.3120 & \cellcolor{green!10} 0.3543 & 0.3489 & 0.3514 & \cellcolor{green!25} 0.3549 & \cellcolor{green!40} 0.3550 \\
rte & 0.5415 & 0.5221 & 0.5176 & \cellcolor{green!25} 0.5469 & \cellcolor{green!40} 0.5560 & \cellcolor{green!10} 0.5266 \\
winogrande & 0.6938 & 0.6862 & \cellcolor{green!10} 0.6873 & \cellcolor{green!25} 0.6893 & 0.6871 & \cellcolor{green!40} 0.6947 \\
Mean & 0.5742 & 0.5811 & 0.5792 & 0.5817 & 0.5852 & 0.5835 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.427474 | 0.381853 |        0.428274 |
| arc_easy      |   0.744949 | 0.662234 |        0.715251 |
| boolq         |   0.74159  | 0.693502 |        0.729549 |
| hellaswag     |   0.558156 | 0.563175 |        0.634977 |
| openbookqa    |   0.312    | 0.3195   |        0.355    |
| rte           |   0.541516 | 0.548511 |        0.526625 |
| winogrande    |   0.693765 | 0.661109 |        0.694653 |
| Mean          |   0.574207 | 0.547126 |        0.583476 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.3819 & \cellcolor{blue!15} \textbf{0.4283} \\
arc_easy & 0.7449 & 0.6622 & \cellcolor{blue!15} \textbf{0.7153} \\
boolq & 0.7416 & 0.6935 & \cellcolor{blue!15} \textbf{0.7295} \\
hellaswag & 0.5582 & 0.5632 & \cellcolor{blue!15} \textbf{0.6350} \\
openbookqa & 0.3120 & 0.3195 & \cellcolor{blue!15} \textbf{0.3550} \\
rte & 0.5415 & \cellcolor{blue!15} \textbf{0.5485} & 0.5266 \\
winogrande & 0.6938 & 0.6611 & \cellcolor{blue!15} \textbf{0.6947} \\
Mean & 0.5742 & 0.5471 & 0.5835 \\
\bottomrule
\end{tabular}
```
