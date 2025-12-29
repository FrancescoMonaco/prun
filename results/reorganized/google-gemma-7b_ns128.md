# Results for google/gemma-7b (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | most_similar   | random     |
|:--------------|-----------:|:------------|:---------------|:-----------|
| arc_challenge |     0.4983 | **0.3763**  | **0.3727**     | **0.3829** |
| arc_easy      |     0.8262 | **0.8007**  | **0.7996**     | **0.8041** |
| boolq         |     0.8361 | **0.6555**  | **0.6379**     | **0.6567** |
| hellaswag     |     0.6066 | **0.6912**  | **0.6908**     | **0.6956** |
| openbookqa    |     0.32   | **0.3850**  | **0.3815**     | **0.3862** |
| rte           |     0.6787 | **0.5523**  | **0.5542**     | **0.6137** |
| winogrande    |     0.7537 | **0.6382**  | **0.6347**     | **0.6464** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrr}
\toprule
 & original & decoupled & most_similar & random \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!25} 0.3763 & \cellcolor{green!10} 0.3727 & \cellcolor{green!40} 0.3829 \\
arc_easy & 0.8262 & \cellcolor{green!25} 0.8007 & \cellcolor{green!10} 0.7996 & \cellcolor{green!40} 0.8041 \\
boolq & 0.8361 & \cellcolor{green!25} 0.6555 & \cellcolor{green!10} 0.6379 & \cellcolor{green!40} 0.6567 \\
hellaswag & 0.6066 & \cellcolor{green!25} 0.6912 & \cellcolor{green!10} 0.6908 & \cellcolor{green!40} 0.6956 \\
openbookqa & 0.3200 & \cellcolor{green!25} 0.3850 & \cellcolor{green!10} 0.3815 & \cellcolor{green!40} 0.3862 \\
rte & 0.6787 & \cellcolor{green!10} 0.5523 & \cellcolor{green!25} 0.5542 & \cellcolor{green!40} 0.6137 \\
winogrande & 0.7537 & \cellcolor{green!25} 0.6382 & \cellcolor{green!10} 0.6347 & \cellcolor{green!40} 0.6464 \\
\bottomrule
\end{tabular}
```
