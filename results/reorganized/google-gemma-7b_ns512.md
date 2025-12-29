# Results for google/gemma-7b (nsamples=512)

## Average across Calibration Groups

| task          |   original | decoupled   | most_similar   | random     |
|:--------------|-----------:|:------------|:---------------|:-----------|
| arc_challenge |     0.4983 | **0.3785**  | **0.3735**     | **0.3837** |
| arc_easy      |     0.8262 | **0.8000**  | **0.8011**     | **0.8027** |
| boolq         |     0.8361 | **0.6528**  | **0.6478**     | **0.6570** |
| hellaswag     |     0.6066 | **0.6907**  | **0.6907**     | **0.6964** |
| openbookqa    |     0.32   | **0.3835**  | **0.3825**     | **0.3870** |
| rte           |     0.6787 | **0.5451**  | **0.5433**     | **0.6092** |
| winogrande    |     0.7537 | **0.6346**  | **0.6343**     | **0.6420** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrr}
\toprule
 & original & decoupled & most_similar & random \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!25} 0.3785 & \cellcolor{green!10} 0.3735 & \cellcolor{green!40} 0.3837 \\
arc_easy & 0.8262 & \cellcolor{green!10} 0.8000 & \cellcolor{green!25} 0.8011 & \cellcolor{green!40} 0.8027 \\
boolq & 0.8361 & \cellcolor{green!25} 0.6528 & \cellcolor{green!10} 0.6478 & \cellcolor{green!40} 0.6570 \\
hellaswag & 0.6066 & \cellcolor{green!10} 0.6907 & \cellcolor{green!25} 0.6907 & \cellcolor{green!40} 0.6964 \\
openbookqa & 0.3200 & \cellcolor{green!25} 0.3835 & \cellcolor{green!10} 0.3825 & \cellcolor{green!40} 0.3870 \\
rte & 0.6787 & \cellcolor{green!25} 0.5451 & \cellcolor{green!10} 0.5433 & \cellcolor{green!40} 0.6092 \\
winogrande & 0.7537 & \cellcolor{green!25} 0.6346 & \cellcolor{green!10} 0.6343 & \cellcolor{green!40} 0.6420 \\
\bottomrule
\end{tabular}
```
