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

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.498294 | 0.444246 |        0.394678 |
| arc_easy      |   0.826178 | 0.726786 |        0.693445 |
| boolq         |   0.836086 | 0.725937 |        0.684213 |
| hellaswag     |   0.606552 | 0.588015 |        0.506933 |
| openbookqa    |   0.32     | 0.345062 |        0.31625  |
| rte           |   0.6787   | 0.555505 |        0.478791 |
| winogrande    |   0.753749 | 0.680989 |        0.646606 |
| Mean          |   0.645651 | 0.580934 |        0.531559 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.4442} & 0.3947 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.7268} & 0.6934 \\
boolq & 0.8361 & \cellcolor{blue!15} \textbf{0.7259} & 0.6842 \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.5880} & 0.5069 \\
openbookqa & 0.3200 & \cellcolor{blue!15} \textbf{0.3451} & 0.3163 \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.5555} & 0.4788 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.6810} & 0.6466 \\
Mean & 0.6457 & 0.5809 & 0.5316 \\
\bottomrule
\end{tabular}
```
