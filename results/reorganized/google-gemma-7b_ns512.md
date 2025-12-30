# Results for google/gemma-7b (nsamples=512)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    |   most_similar | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|---------------:|:-----------|
| arc_challenge |     0.4983 | 0.3785      | **0.5156**              | **0.5122** |         0.3735 | **0.3837** |
| arc_easy      |     0.8262 | 0.8000      | **0.8015**              | **0.8021** |         0.8011 | **0.8027** |
| boolq         |     0.8361 | 0.6528      | **0.8232**              | **0.8183** |         0.6478 | **0.6570** |
| hellaswag     |     0.6066 | 0.6907      | **0.6963**              | **0.6928** |         0.6907 | **0.6964** |
| openbookqa    |     0.32   | **0.3835**  | **0.3895**              | 0.3800     |         0.3825 | **0.3870** |
| rte           |     0.6787 | 0.5451      | **0.5921**              | **0.5632** |         0.5433 | **0.6092** |
| winogrande    |     0.7537 | 0.6346      | **0.7368**              | **0.7344** |         0.6343 | **0.6420** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & most_similar & random \\
\midrule
arc_challenge & 0.4983 & 0.3785 & \cellcolor{green!40} 0.5156 & \cellcolor{green!25} 0.5122 & 0.3735 & \cellcolor{green!10} 0.3837 \\
arc_easy & 0.8262 & 0.8000 & \cellcolor{green!10} 0.8015 & \cellcolor{green!25} 0.8021 & 0.8011 & \cellcolor{green!40} 0.8027 \\
boolq & 0.8361 & 0.6528 & \cellcolor{green!40} 0.8232 & \cellcolor{green!25} 0.8183 & 0.6478 & \cellcolor{green!10} 0.6570 \\
hellaswag & 0.6066 & 0.6907 & \cellcolor{green!25} 0.6963 & \cellcolor{green!10} 0.6928 & 0.6907 & \cellcolor{green!40} 0.6964 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3835 & \cellcolor{green!40} 0.3895 & 0.3800 & 0.3825 & \cellcolor{green!25} 0.3870 \\
rte & 0.6787 & 0.5451 & \cellcolor{green!25} 0.5921 & \cellcolor{green!10} 0.5632 & 0.5433 & \cellcolor{green!40} 0.6092 \\
winogrande & 0.7537 & 0.6346 & \cellcolor{green!40} 0.7368 & \cellcolor{green!25} 0.7344 & 0.6343 & \cellcolor{green!10} 0.6420 \\
\bottomrule
\end{tabular}
```
