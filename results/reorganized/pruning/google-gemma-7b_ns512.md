# Results for google/gemma-7b (nsamples=512) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.3795                  | **0.3837**         | 0.3791         | **0.3813** | **0.3947**      |
| arc_easy      |     0.8262 | 0.6778                  | **0.6808**         | 0.6761         | **0.6787** | **0.6934**      |
| boolq         |     0.8361 | **0.6731**              | 0.6701             | **0.6733**     | 0.6689     | **0.6842**      |
| hellaswag     |     0.6066 | **0.4834**              | **0.4903**         | 0.4822         | 0.4832     | **0.5069**      |
| openbookqa    |     0.32   | **0.3121**              | 0.3069             | 0.3084         | **0.3103** | **0.3163**      |
| rte           |     0.6787 | **0.4874**              | **0.5041**         | **0.4810**     | 0.4756     | 0.4788          |
| winogrande    |     0.7537 | 0.6350                  | 0.6378             | **0.6381**     | **0.6398** | **0.6466**      |
| Mean          |     0.6457 | 0.5212                  | 0.5248             | 0.5197         | 0.5197     | 0.5316          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.3795 & \cellcolor{green!25} 0.3837 & 0.3791 & \cellcolor{green!10} 0.3813 & \cellcolor{green!40} 0.3947 \\
arc_easy & 0.8262 & 0.6778 & \cellcolor{green!25} 0.6808 & 0.6761 & \cellcolor{green!10} 0.6787 & \cellcolor{green!40} 0.6934 \\
boolq & 0.8361 & \cellcolor{green!10} 0.6731 & 0.6701 & \cellcolor{green!25} 0.6733 & 0.6689 & \cellcolor{green!40} 0.6842 \\
hellaswag & 0.6066 & \cellcolor{green!10} 0.4834 & \cellcolor{green!25} 0.4903 & 0.4822 & 0.4832 & \cellcolor{green!40} 0.5069 \\
openbookqa & 0.3200 & \cellcolor{green!25} 0.3121 & 0.3069 & 0.3084 & \cellcolor{green!10} 0.3103 & \cellcolor{green!40} 0.3163 \\
rte & 0.6787 & \cellcolor{green!25} 0.4874 & \cellcolor{green!40} 0.5041 & \cellcolor{green!10} 0.4810 & 0.4756 & 0.4788 \\
winogrande & 0.7537 & 0.6350 & 0.6378 & \cellcolor{green!10} 0.6381 & \cellcolor{green!25} 0.6398 & \cellcolor{green!40} 0.6466 \\
Mean & 0.6457 & 0.5212 & 0.5248 & 0.5197 & 0.5197 & 0.5316 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.379124 |
| arc_easy      |   0.826178 |                0.67547  |
| boolq         |   0.836086 |                0.67105  |
| hellaswag     |   0.606552 |                0.484208 |
| openbookqa    |   0.32     |                0.310333 |
| rte           |   0.6787   |                0.487365 |
| winogrande    |   0.753749 |                0.633123 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.3791 \\
arc_easy & 0.8262 & 0.6755 \\
boolq & 0.8361 & 0.6710 \\
hellaswag & 0.6066 & 0.4842 \\
openbookqa & 0.3200 & 0.3103 \\
rte & 0.6787 & 0.4874 \\
winogrande & 0.7537 & 0.6331 \\
\bottomrule
\end{tabular}
```
