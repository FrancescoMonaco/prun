# Results for google/gemma-7b (nsamples=128.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4983 | **0.5169** |
| arc_easy      |     0.8262 | **0.8028** |
| boolq         |     0.8361 | **0.8210** |
| hellaswag     |     0.6066 | **0.6971** |
| openbookqa    |     0.32   | **0.3860** |
| rte           |     0.6787 | **0.6113** |
| winogrande    |     0.7537 | **0.7411** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!40} 0.5169 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.8028 \\
boolq & 0.8361 & \cellcolor{green!40} 0.8210 \\
hellaswag & 0.6066 & \cellcolor{green!40} 0.6971 \\
openbookqa & 0.3200 & \cellcolor{green!40} 0.3860 \\
rte & 0.6787 & \cellcolor{green!40} 0.6113 \\
winogrande & 0.7537 & \cellcolor{green!40} 0.7411 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.498294 | 0.516638 |
| arc_easy      |   0.826178 | 0.807379 |
| boolq         |   0.836086 | 0.811417 |
| hellaswag     |   0.606552 | 0.698632 |
| openbookqa    |   0.32     | 0.386    |
| rte           |   0.6787   | 0.672684 |
| winogrande    |   0.753749 | 0.74191  |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4983 & 0.5166 \\
arc_easy & 0.8262 & 0.8074 \\
boolq & 0.8361 & 0.8114 \\
hellaswag & 0.6066 & 0.6986 \\
openbookqa & 0.3200 & 0.3860 \\
rte & 0.6787 & 0.6727 \\
winogrande & 0.7537 & 0.7419 \\
\bottomrule
\end{tabular}
```
