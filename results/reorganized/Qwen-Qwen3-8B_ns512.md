# Results for Qwen/Qwen3-8B (nsamples=512)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | distribution_matching_no_outliers   | herding    | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:------------------------------------|:-----------|:------------------|:---------------|:-----------|
| arc_challenge |     0.558  | 0.4177      | **0.5574**              | **0.5589**                          | **0.5552** | 0.4194            | 0.4157         | 0.4181     |
| arc_easy      |     0.8359 | **0.8192**  | 0.8165                  | 0.8142                              | 0.8145     | **0.8195**        | **0.8195**     | 0.8148     |
| boolq         |     0.8661 | 0.7486      | **0.8647**              | **0.8641**                          | **0.8654** | 0.7512            | 0.7497         | 0.7498     |
| hellaswag     |     0.5713 | 0.6493      | **0.6526**              | **0.6532**                          | 0.6497     | 0.6502            | 0.6502         | **0.6526** |
| openbookqa    |     0.31   | 0.3640      | **0.3695**              | 0.3670                              | **0.3690** | 0.3650            | 0.3650         | **0.3695** |
| rte           |     0.7834 | 0.7599      | **0.7635**              | 0.7581                              | **0.7653** | 0.7617            | 0.7617         | **0.7653** |
| winogrande    |     0.6772 | 0.6230      | **0.6851**              | **0.6855**                          | **0.6914** | 0.6239            | 0.6235         | 0.6270     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.5580 & 0.4177 & \cellcolor{green!25} 0.5574 & \cellcolor{green!40} 0.5589 & \cellcolor{green!10} 0.5552 & 0.4194 & 0.4157 & 0.4181 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.8192 & 0.8165 & 0.8142 & 0.8145 & \cellcolor{green!40} 0.8195 & \cellcolor{green!25} 0.8195 & 0.8148 \\
boolq & 0.8661 & 0.7486 & \cellcolor{green!25} 0.8647 & \cellcolor{green!10} 0.8641 & \cellcolor{green!40} 0.8654 & 0.7512 & 0.7497 & 0.7498 \\
hellaswag & 0.5713 & 0.6493 & \cellcolor{green!10} 0.6526 & \cellcolor{green!40} 0.6532 & 0.6497 & 0.6502 & 0.6502 & \cellcolor{green!25} 0.6526 \\
openbookqa & 0.3100 & 0.3640 & \cellcolor{green!40} 0.3695 & 0.3670 & \cellcolor{green!10} 0.3690 & 0.3650 & 0.3650 & \cellcolor{green!25} 0.3695 \\
rte & 0.7834 & 0.7599 & \cellcolor{green!10} 0.7635 & 0.7581 & \cellcolor{green!40} 0.7653 & 0.7617 & 0.7617 & \cellcolor{green!25} 0.7653 \\
winogrande & 0.6772 & 0.6230 & \cellcolor{green!10} 0.6851 & \cellcolor{green!25} 0.6855 & \cellcolor{green!40} 0.6914 & 0.6239 & 0.6235 & 0.6270 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.55802  | 0.553328 |                0.557381 |
| arc_easy      |   0.835859 | 0.819935 |                0.816498 |
| boolq         |   0.866055 | 0.863303 |                0.864679 |
| hellaswag     |   0.571301 | 0.649622 |                0.652559 |
| openbookqa    |   0.31     | 0.363667 |                0.3695   |
| rte           |   0.783394 | 0.766546 |                0.763538 |
| winogrande    |   0.67719  | 0.68061  |                0.685083 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.5580 & 0.5533 & \cellcolor{blue!15} \textbf{0.5574} \\
arc_easy & 0.8359 & \cellcolor{blue!15} \textbf{0.8199} & 0.8165 \\
boolq & 0.8661 & 0.8633 & \cellcolor{blue!15} \textbf{0.8647} \\
hellaswag & 0.5713 & 0.6496 & \cellcolor{blue!15} \textbf{0.6526} \\
openbookqa & 0.3100 & 0.3637 & \cellcolor{blue!15} \textbf{0.3695} \\
rte & 0.7834 & \cellcolor{blue!15} \textbf{0.7665} & 0.7635 \\
winogrande & 0.6772 & 0.6806 & \cellcolor{blue!15} \textbf{0.6851} \\
\bottomrule
\end{tabular}
```
