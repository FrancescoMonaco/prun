# Results for google/gemma-7b (nsamples=1024) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.4933                  | **0.4941**         | 0.4898         | **0.4941** | **0.4948**      |
| arc_easy      |     0.8262 | **0.7665**              | 0.7644             | 0.7559         | **0.7674** | **0.7675**      |
| boolq         |     0.8361 | **0.7661**              | **0.7604**         | 0.7557         | **0.7669** | 0.7576          |
| hellaswag     |     0.6066 | **0.6678**              | **0.6654**         | 0.6639         | 0.6641     | **0.6701**      |
| openbookqa    |     0.32   | **0.3838**              | **0.3772**         | **0.3765**     | 0.3745     | 0.3742          |
| rte           |     0.6787 | **0.6440**              | 0.6236             | **0.6331**     | 0.6264     | **0.6295**      |
| winogrande    |     0.7537 | 0.7196                  | 0.7146             | **0.7206**     | **0.7258** | **0.7246**      |
| Mean          |     0.6457 | 0.6344                  | 0.6285             | 0.6279         | 0.6313     | 0.6312          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4933 & \cellcolor{green!10} 0.4941 & 0.4898 & \cellcolor{green!25} 0.4941 & \cellcolor{green!40} 0.4948 \\
arc_easy & 0.8262 & \cellcolor{green!10} 0.7665 & 0.7644 & 0.7559 & \cellcolor{green!25} 0.7674 & \cellcolor{green!40} 0.7675 \\
boolq & 0.8361 & \cellcolor{green!25} 0.7661 & \cellcolor{green!10} 0.7604 & 0.7557 & \cellcolor{green!40} 0.7669 & 0.7576 \\
hellaswag & 0.6066 & \cellcolor{green!25} 0.6678 & \cellcolor{green!10} 0.6654 & 0.6639 & 0.6641 & \cellcolor{green!40} 0.6701 \\
openbookqa & 0.3200 & \cellcolor{green!40} 0.3838 & \cellcolor{green!25} 0.3772 & \cellcolor{green!10} 0.3765 & 0.3745 & 0.3742 \\
rte & 0.6787 & \cellcolor{green!40} 0.6440 & 0.6236 & \cellcolor{green!25} 0.6331 & 0.6264 & \cellcolor{green!10} 0.6295 \\
winogrande & 0.7537 & 0.7196 & 0.7146 & \cellcolor{green!10} 0.7206 & \cellcolor{green!40} 0.7258 & \cellcolor{green!25} 0.7246 \\
Mean & 0.6457 & 0.6344 & 0.6285 & 0.6279 & 0.6313 & 0.6312 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.492605 |
| arc_easy      |   0.826178 |                0.763258 |
| boolq         |   0.836086 |                0.762487 |
| hellaswag     |   0.606552 |                0.666899 |
| openbookqa    |   0.32     |                0.382667 |
| rte           |   0.6787   |                0.645608 |
| winogrande    |   0.753749 |                0.720205 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.4926 \\
arc_easy & 0.8262 & 0.7633 \\
boolq & 0.8361 & 0.7625 \\
hellaswag & 0.6066 & 0.6669 \\
openbookqa & 0.3200 & 0.3827 \\
rte & 0.6787 & 0.6456 \\
winogrande & 0.7537 & 0.7202 \\
\bottomrule
\end{tabular}
```
