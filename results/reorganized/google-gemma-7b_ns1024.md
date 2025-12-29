# Results for google/gemma-7b (nsamples=1024)

## Average across Calibration Groups

| task          |   original | decoupled   | most_similar   | random     |
|:--------------|-----------:|:------------|:---------------|:-----------|
| arc_challenge |     0.4983 | **0.3726**  | **0.3671**     | **0.3808** |
| arc_easy      |     0.8262 | **0.8044**  | **0.8019**     | **0.8044** |
| boolq         |     0.8361 | **0.6459**  | **0.6437**     | **0.6578** |
| hellaswag     |     0.6066 | **0.6914**  | **0.6909**     | **0.6957** |
| openbookqa    |     0.32   | **0.3820**  | **0.3830**     | **0.3853** |
| rte           |     0.6787 | **0.5439**  | **0.5439**     | **0.6173** |
| winogrande    |     0.7537 | **0.6294**  | **0.6233**     | **0.6384** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrr}
\toprule
 & original & decoupled & most_similar & random \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!25} 0.3726 & \cellcolor{green!10} 0.3671 & \cellcolor{green!40} 0.3808 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.8044 & \cellcolor{green!10} 0.8019 & \cellcolor{green!25} 0.8044 \\
boolq & 0.8361 & \cellcolor{green!25} 0.6459 & \cellcolor{green!10} 0.6437 & \cellcolor{green!40} 0.6578 \\
hellaswag & 0.6066 & \cellcolor{green!25} 0.6914 & \cellcolor{green!10} 0.6909 & \cellcolor{green!40} 0.6957 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3820 & \cellcolor{green!25} 0.3830 & \cellcolor{green!40} 0.3853 \\
rte & 0.6787 & \cellcolor{green!25} 0.5439 & \cellcolor{green!10} 0.5439 & \cellcolor{green!40} 0.6173 \\
winogrande & 0.7537 & \cellcolor{green!25} 0.6294 & \cellcolor{green!10} 0.6233 & \cellcolor{green!40} 0.6384 \\
\bottomrule
\end{tabular}
```
