# Results for meta-llama/Llama-3.2-3B (nsamples=128) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | 0.4185                  | **0.4253**         | **0.4251**     | 0.4112     | **0.4273**      |
| arc_easy      |     0.7449 | **0.7113**              | **0.7087**         | 0.7061         | 0.6955     | **0.7134**      |
| boolq         |     0.7416 | **0.7204**              | **0.7171**         | 0.6965         | 0.7164     | **0.7256**      |
| hellaswag     |     0.5582 | 0.6284                  | 0.6292             | **0.6299**     | **0.6293** | **0.6308**      |
| openbookqa    |     0.312  | 0.3509                  | 0.3510             | **0.3561**     | **0.3565** | **0.3590**      |
| rte           |     0.5415 | 0.5190                  | **0.5447**         | **0.5235**     | **0.5560** | 0.5217          |
| winogrande    |     0.6938 | 0.6799                  | **0.6878**         | 0.6834         | **0.6841** | **0.6892**      |
| Mean          |     0.5742 | 0.5754                  | 0.5805             | 0.5744         | 0.5784     | 0.5810          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.4185 & \cellcolor{green!25} 0.4253 & \cellcolor{green!10} 0.4251 & 0.4112 & \cellcolor{green!40} 0.4273 \\
arc_easy & 0.7449 & \cellcolor{green!25} 0.7113 & \cellcolor{green!10} 0.7087 & 0.7061 & 0.6955 & \cellcolor{green!40} 0.7134 \\
boolq & 0.7416 & \cellcolor{green!25} 0.7204 & \cellcolor{green!10} 0.7171 & 0.6965 & 0.7164 & \cellcolor{green!40} 0.7256 \\
hellaswag & 0.5582 & 0.6284 & 0.6292 & \cellcolor{green!25} 0.6299 & \cellcolor{green!10} 0.6293 & \cellcolor{green!40} 0.6308 \\
openbookqa & 0.3120 & 0.3509 & 0.3510 & \cellcolor{green!10} 0.3561 & \cellcolor{green!25} 0.3565 & \cellcolor{green!40} 0.3590 \\
rte & 0.5415 & 0.5190 & \cellcolor{green!25} 0.5447 & \cellcolor{green!10} 0.5235 & \cellcolor{green!40} 0.5560 & 0.5217 \\
winogrande & 0.6938 & 0.6799 & \cellcolor{green!25} 0.6878 & 0.6834 & \cellcolor{green!10} 0.6841 & \cellcolor{green!40} 0.6892 \\
Mean & 0.5742 & 0.5754 & 0.5805 & 0.5744 & 0.5784 & 0.5810 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.427474 | 0.375178 |        0.427314 |
| arc_easy      |   0.744949 | 0.654242 |        0.713436 |
| boolq         |   0.74159  | 0.671885 |        0.725573 |
| hellaswag     |   0.558156 | 0.562701 |        0.630832 |
| openbookqa    |   0.312    | 0.320958 |        0.359    |
| rte           |   0.541516 | 0.536552 |        0.521661 |
| winogrande    |   0.693765 | 0.666732 |        0.689227 |
| Mean          |   0.574207 | 0.541178 |        0.581006 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.3752 & \cellcolor{blue!15} \textbf{0.4273} \\
arc_easy & 0.7449 & 0.6542 & \cellcolor{blue!15} \textbf{0.7134} \\
boolq & 0.7416 & 0.6719 & \cellcolor{blue!15} \textbf{0.7256} \\
hellaswag & 0.5582 & 0.5627 & \cellcolor{blue!15} \textbf{0.6308} \\
openbookqa & 0.3120 & 0.3210 & \cellcolor{blue!15} \textbf{0.3590} \\
rte & 0.5415 & \cellcolor{blue!15} \textbf{0.5366} & 0.5217 \\
winogrande & 0.6938 & 0.6667 & \cellcolor{blue!15} \textbf{0.6892} \\
Mean & 0.5742 & 0.5412 & 0.5810 \\
\bottomrule
\end{tabular}
```
