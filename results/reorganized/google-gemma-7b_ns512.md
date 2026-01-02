# Results for google/gemma-7b (nsamples=512)

## Average across Calibration Groups

| task          |   original |   decoupled | distribution_matching   | distribution_matching_no_outliers   | herding    |   most_similar | random     |
|:--------------|-----------:|------------:|:------------------------|:------------------------------------|:-----------|---------------:|:-----------|
| arc_challenge |     0.4983 |      0.3785 | **0.5156**              | **0.5171**                          | **0.5122** |         0.3735 | 0.3837     |
| arc_easy      |     0.8262 |      0.8    | 0.8015                  | **0.8041**                          | **0.8021** |         0.8011 | **0.8027** |
| boolq         |     0.8361 |      0.6528 | **0.8232**              | **0.8271**                          | **0.8183** |         0.6478 | 0.6570     |
| hellaswag     |     0.6066 |      0.6907 | **0.6963**              | **0.6958**                          | 0.6928     |         0.6907 | **0.6964** |
| openbookqa    |     0.32   |      0.3835 | **0.3895**              | **0.3910**                          | 0.3800     |         0.3825 | **0.3870** |
| rte           |     0.6787 |      0.5451 | **0.5921**              | **0.6047**                          | 0.5632     |         0.5433 | **0.6092** |
| winogrande    |     0.7537 |      0.6346 | **0.7368**              | **0.7380**                          | **0.7344** |         0.6343 | 0.6420     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_similar & random \\
\midrule
arc_challenge & 0.4983 & 0.3785 & \cellcolor{green!25} 0.5156 & \cellcolor{green!40} 0.5171 & \cellcolor{green!10} 0.5122 & 0.3735 & 0.3837 \\
arc_easy & 0.8262 & 0.8000 & 0.8015 & \cellcolor{green!40} 0.8041 & \cellcolor{green!10} 0.8021 & 0.8011 & \cellcolor{green!25} 0.8027 \\
boolq & 0.8361 & 0.6528 & \cellcolor{green!25} 0.8232 & \cellcolor{green!40} 0.8271 & \cellcolor{green!10} 0.8183 & 0.6478 & 0.6570 \\
hellaswag & 0.6066 & 0.6907 & \cellcolor{green!25} 0.6963 & \cellcolor{green!10} 0.6958 & 0.6928 & 0.6907 & \cellcolor{green!40} 0.6964 \\
openbookqa & 0.3200 & 0.3835 & \cellcolor{green!25} 0.3895 & \cellcolor{green!40} 0.3910 & 0.3800 & 0.3825 & \cellcolor{green!10} 0.3870 \\
rte & 0.6787 & 0.5451 & \cellcolor{green!10} 0.5921 & \cellcolor{green!25} 0.6047 & 0.5632 & 0.5433 & \cellcolor{green!40} 0.6092 \\
winogrande & 0.7537 & 0.6346 & \cellcolor{green!25} 0.7368 & \cellcolor{green!40} 0.7380 & \cellcolor{green!10} 0.7344 & 0.6343 & 0.6420 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.498294 | 0.518203 |                0.515572 |
| arc_easy      |   0.826178 | 0.807029 |                0.801452 |
| boolq         |   0.836086 | 0.813456 |                0.823242 |
| hellaswag     |   0.606552 | 0.698981 |                0.696276 |
| openbookqa    |   0.32     | 0.387333 |                0.3895   |
| rte           |   0.6787   | 0.67509  |                0.592058 |
| winogrande    |   0.753749 | 0.743225 |                0.73678  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.5182} & 0.5156 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.8070} & 0.8015 \\
boolq & 0.8361 & 0.8135 & \cellcolor{blue!15} \textbf{0.8232} \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.6990} & 0.6963 \\
openbookqa & 0.3200 & 0.3873 & \cellcolor{blue!15} \textbf{0.3895} \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.6751} & 0.5921 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.7432} & 0.7368 \\
\bottomrule
\end{tabular}
```
