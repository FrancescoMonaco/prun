# Results for google/gemma-7b (nsamples=1024.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4983 | **0.5132** |
| arc_easy      |     0.8262 | **0.8035** |
| boolq         |     0.8361 | **0.8225** |
| hellaswag     |     0.6066 | **0.6958** |
| openbookqa    |     0.32   | **0.3860** |
| rte           |     0.6787 | **0.5903** |
| winogrande    |     0.7537 | **0.7399** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!40} 0.5132 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.8035 \\
boolq & 0.8361 & \cellcolor{green!40} 0.8225 \\
hellaswag & 0.6066 & \cellcolor{green!40} 0.6958 \\
openbookqa & 0.3200 & \cellcolor{green!40} 0.3860 \\
rte & 0.6787 & \cellcolor{green!40} 0.5903 \\
winogrande & 0.7537 & \cellcolor{green!40} 0.7399 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.498294 | 0.517065 |
| arc_easy      |   0.826178 | 0.807379 |
| boolq         |   0.836086 | 0.811621 |
| hellaswag     |   0.606552 | 0.698898 |
| openbookqa    |   0.32     | 0.387333 |
| rte           |   0.6787   | 0.672684 |
| winogrande    |   0.753749 | 0.743225 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4983 & 0.5171 \\
arc_easy & 0.8262 & 0.8074 \\
boolq & 0.8361 & 0.8116 \\
hellaswag & 0.6066 & 0.6989 \\
openbookqa & 0.3200 & 0.3873 \\
rte & 0.6787 & 0.6727 \\
winogrande & 0.7537 & 0.7432 \\
\bottomrule
\end{tabular}
```
