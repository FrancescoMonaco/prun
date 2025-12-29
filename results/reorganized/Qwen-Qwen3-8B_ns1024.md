# Results for Qwen/Qwen3-8B (nsamples=1024)

## Average across Calibration Groups

| task          |   original | decoupled   | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------|:---------------|:-----------|
| arc_challenge |     0.558  | **0.4661**  | **0.4632**        | 0.4589         | **0.4630** |
| arc_easy      |     0.8359 | **0.8170**  | **0.8189**        | **0.8189**     | 0.8162     |
| boolq         |     0.8661 | **0.7846**  | **0.7816**        | 0.7793         | **0.7831** |
| hellaswag     |     0.5713 | 0.6498      | **0.6505**        | **0.6505**     | **0.6528** |
| openbookqa    |     0.31   | 0.3600      | **0.3640**        | **0.3640**     | **0.3685** |
| rte           |     0.7834 | **0.7617**  | **0.7690**        | **0.7690**     | 0.7545     |
| winogrande    |     0.6772 | 0.6525      | **0.6545**        | **0.6544**     | **0.6547** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrr}
\toprule
 & original & decoupled & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.5580 & \cellcolor{green!40} 0.4661 & \cellcolor{green!25} 0.4632 & 0.4589 & \cellcolor{green!10} 0.4630 \\
arc_easy & 0.8359 & \cellcolor{green!10} 0.8170 & \cellcolor{green!40} 0.8189 & \cellcolor{green!25} 0.8189 & 0.8162 \\
boolq & 0.8661 & \cellcolor{green!40} 0.7846 & \cellcolor{green!10} 0.7816 & 0.7793 & \cellcolor{green!25} 0.7831 \\
hellaswag & 0.5713 & 0.6498 & \cellcolor{green!25} 0.6505 & \cellcolor{green!10} 0.6505 & \cellcolor{green!40} 0.6528 \\
openbookqa & 0.3100 & 0.3600 & \cellcolor{green!25} 0.3640 & \cellcolor{green!10} 0.3640 & \cellcolor{green!40} 0.3685 \\
rte & 0.7834 & \cellcolor{green!10} 0.7617 & \cellcolor{green!40} 0.7690 & \cellcolor{green!25} 0.7690 & 0.7545 \\
winogrande & 0.6772 & 0.6525 & \cellcolor{green!25} 0.6545 & \cellcolor{green!10} 0.6544 & \cellcolor{green!40} 0.6547 \\
\bottomrule
\end{tabular}
```
