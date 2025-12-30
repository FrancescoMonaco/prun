# Results for Qwen/Qwen3-8B (nsamples=512)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|:------------------|:---------------|:-----------|
| arc_challenge |     0.558  | 0.4177      | **0.5574**              | **0.5552** | **0.4194**        | 0.4157         | 0.4181     |
| arc_easy      |     0.8359 | **0.8192**  | 0.8165                  | 0.8145     | **0.8195**        | **0.8195**     | 0.8148     |
| boolq         |     0.8661 | 0.7486      | **0.8647**              | **0.8654** | **0.7512**        | 0.7497         | 0.7498     |
| hellaswag     |     0.5713 | 0.6493      | **0.6526**              | 0.6497     | **0.6502**        | 0.6502         | **0.6526** |
| openbookqa    |     0.31   | 0.3640      | **0.3695**              | **0.3690** | 0.3650            | 0.3650         | **0.3695** |
| rte           |     0.7834 | 0.7599      | **0.7635**              | **0.7653** | 0.7617            | 0.7617         | **0.7653** |
| winogrande    |     0.6772 | 0.6230      | **0.6851**              | **0.6914** | 0.6239            | 0.6235         | **0.6270** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.5580 & 0.4177 & \cellcolor{green!40} 0.5574 & \cellcolor{green!25} 0.5552 & \cellcolor{green!10} 0.4194 & 0.4157 & 0.4181 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.8192 & 0.8165 & 0.8145 & \cellcolor{green!40} 0.8195 & \cellcolor{green!25} 0.8195 & 0.8148 \\
boolq & 0.8661 & 0.7486 & \cellcolor{green!25} 0.8647 & \cellcolor{green!40} 0.8654 & \cellcolor{green!10} 0.7512 & 0.7497 & 0.7498 \\
hellaswag & 0.5713 & 0.6493 & \cellcolor{green!25} 0.6526 & 0.6497 & \cellcolor{green!10} 0.6502 & 0.6502 & \cellcolor{green!40} 0.6526 \\
openbookqa & 0.3100 & 0.3640 & \cellcolor{green!40} 0.3695 & \cellcolor{green!10} 0.3690 & 0.3650 & 0.3650 & \cellcolor{green!25} 0.3695 \\
rte & 0.7834 & 0.7599 & \cellcolor{green!10} 0.7635 & \cellcolor{green!40} 0.7653 & 0.7617 & 0.7617 & \cellcolor{green!25} 0.7653 \\
winogrande & 0.6772 & 0.6230 & \cellcolor{green!25} 0.6851 & \cellcolor{green!40} 0.6914 & 0.6239 & 0.6235 & \cellcolor{green!10} 0.6270 \\
\bottomrule
\end{tabular}
```
