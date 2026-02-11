# Results for google/gemma-7b (nsamples=1024) - quantization

## Average across Calibration Groups

| task          |   original |   dictionary | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|-------------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 |       0.3799 | 0.4933                  | **0.4941**         | 0.4898         | **0.4941** | **0.4948**      |
| arc_easy      |     0.8262 |       0.6056 | **0.7665**              | 0.7644             | 0.7559         | **0.7674** | **0.7675**      |
| boolq         |     0.8361 |       0.6875 | **0.7661**              | **0.7604**         | 0.7557         | **0.7669** | 0.7576          |
| hellaswag     |     0.6066 |       0.496  | **0.6678**              | **0.6654**         | 0.6639         | 0.6641     | **0.6701**      |
| openbookqa    |     0.32   |       0.3435 | **0.3838**              | **0.3772**         | **0.3765**     | 0.3745     | 0.3742          |
| rte           |     0.6787 |       0.6137 | **0.6440**              | 0.6236             | **0.6331**     | 0.6264     | **0.6295**      |
| winogrande    |     0.7537 |       0.5955 | 0.7196                  | 0.7146             | **0.7206**     | **0.7258** | **0.7246**      |
| Mean          |     0.6457 |       0.5317 | 0.6344                  | 0.6285             | 0.6279         | 0.6313     | 0.6312          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & dictionary & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.3799 & 0.4933 & \cellcolor{green!10} 0.4941 & 0.4898 & \cellcolor{green!25} 0.4941 & \cellcolor{green!40} 0.4948 \\
arc_easy & 0.8262 & 0.6056 & \cellcolor{green!10} 0.7665 & 0.7644 & 0.7559 & \cellcolor{green!25} 0.7674 & \cellcolor{green!40} 0.7675 \\
boolq & 0.8361 & 0.6875 & \cellcolor{green!25} 0.7661 & \cellcolor{green!10} 0.7604 & 0.7557 & \cellcolor{green!40} 0.7669 & 0.7576 \\
hellaswag & 0.6066 & 0.4960 & \cellcolor{green!25} 0.6678 & \cellcolor{green!10} 0.6654 & 0.6639 & 0.6641 & \cellcolor{green!40} 0.6701 \\
openbookqa & 0.3200 & 0.3435 & \cellcolor{green!40} 0.3838 & \cellcolor{green!25} 0.3772 & \cellcolor{green!10} 0.3765 & 0.3745 & 0.3742 \\
rte & 0.6787 & 0.6137 & \cellcolor{green!40} 0.6440 & 0.6236 & \cellcolor{green!25} 0.6331 & 0.6264 & \cellcolor{green!10} 0.6295 \\
winogrande & 0.7537 & 0.5955 & 0.7196 & 0.7146 & \cellcolor{green!10} 0.7206 & \cellcolor{green!40} 0.7258 & \cellcolor{green!25} 0.7246 \\
Mean & 0.6457 & 0.5317 & 0.6344 & 0.6285 & 0.6279 & 0.6313 & 0.6312 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.440833 |        0.494827 |
| arc_easy      |   0.826178 | 0.723143 |        0.767519 |
| boolq         |   0.836086 | 0.721235 |        0.757607 |
| hellaswag     |   0.606552 | 0.582152 |        0.670104 |
| openbookqa    |   0.32     | 0.3395   |        0.37425  |
| rte           |   0.6787   | 0.550993 |        0.629513 |
| winogrande    |   0.753749 | 0.680841 |        0.724645 |
| Mean          |   0.645651 | 0.576957 |        0.631209 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4408 & \cellcolor{blue!15} \textbf{0.4948} \\
arc_easy & 0.8262 & 0.7231 & \cellcolor{blue!15} \textbf{0.7675} \\
boolq & 0.8361 & 0.7212 & \cellcolor{blue!15} \textbf{0.7576} \\
hellaswag & 0.6066 & 0.5822 & \cellcolor{blue!15} \textbf{0.6701} \\
openbookqa & 0.3200 & 0.3395 & \cellcolor{blue!15} \textbf{0.3742} \\
rte & 0.6787 & 0.5510 & \cellcolor{blue!15} \textbf{0.6295} \\
winogrande & 0.7537 & 0.6808 & \cellcolor{blue!15} \textbf{0.7246} \\
Mean & 0.6457 & 0.5770 & 0.6312 \\
\bottomrule
\end{tabular}
```
