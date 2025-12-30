# Results for Qwen/Qwen3-8B (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|:------------------|:---------------|:-----------|
| arc_challenge |     0.558  | **0.4707**  | **0.5574**              | **0.5578** | 0.4655            | 0.4618         | 0.4141     |
| arc_easy      |     0.8359 | 0.8158      | **0.8166**              | 0.8140     | **0.8163**        | **0.8163**     | 0.8162     |
| boolq         |     0.8661 | 0.7982      | **0.8650**              | **0.8624** | **0.8012**        | 0.7996         | 0.7531     |
| hellaswag     |     0.5713 | **0.6513**  | **0.6525**              | 0.6493     | 0.6503            | 0.6503         | **0.6532** |
| openbookqa    |     0.31   | **0.3715**  | **0.3670**              | 0.3645     | 0.3655            | 0.3655         | **0.3690** |
| rte           |     0.7834 | 0.7617      | 0.7545                  | **0.7762** | **0.7780**        | **0.7780**     | 0.7653     |
| winogrande    |     0.6772 | **0.6594**  | **0.6835**              | **0.6819** | 0.6571            | 0.6562         | 0.6214     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!10} 0.4707 & \cellcolor{green!25} 0.5574 & \cellcolor{green!40} 0.5578 & 0.4655 & 0.4618 & 0.4141 \\
arc_easy & 0.8359 & 0.8158 & \cellcolor{green!40} 0.8166 & 0.8140 & \cellcolor{green!25} 0.8163 & \cellcolor{green!10} 0.8163 & 0.8162 \\
boolq & 0.8661 & 0.7982 & \cellcolor{green!40} 0.8650 & \cellcolor{green!25} 0.8624 & \cellcolor{green!10} 0.8012 & 0.7996 & 0.7531 \\
hellaswag & 0.5713 & \cellcolor{green!10} 0.6513 & \cellcolor{green!25} 0.6525 & 0.6493 & 0.6503 & 0.6503 & \cellcolor{green!40} 0.6532 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3715 & \cellcolor{green!10} 0.3670 & 0.3645 & 0.3655 & 0.3655 & \cellcolor{green!25} 0.3690 \\
rte & 0.7834 & 0.7617 & 0.7545 & \cellcolor{green!10} 0.7762 & \cellcolor{green!40} 0.7780 & \cellcolor{green!25} 0.7780 & 0.7653 \\
winogrande & 0.6772 & \cellcolor{green!10} 0.6594 & \cellcolor{green!40} 0.6835 & \cellcolor{green!25} 0.6819 & 0.6571 & 0.6562 & 0.6214 \\
\bottomrule
\end{tabular}
```
