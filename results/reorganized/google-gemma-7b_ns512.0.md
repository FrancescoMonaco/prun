# Results for google/gemma-7b (nsamples=512.0)

## Average across Calibration Groups

| task          |   original | zipf       |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4983 | **0.5152** |
| arc_easy      |     0.8262 | **0.8032** |
| boolq         |     0.8361 | **0.8234** |
| hellaswag     |     0.6066 | **0.6966** |
| openbookqa    |     0.32   | **0.3860** |
| rte           |     0.6787 | **0.6077** |
| winogrande    |     0.7537 | **0.7414** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & zipf \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!40} 0.5152 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.8032 \\
boolq & 0.8361 & \cellcolor{green!40} 0.8234 \\
hellaswag & 0.6066 & \cellcolor{green!40} 0.6966 \\
openbookqa & 0.3200 & \cellcolor{green!40} 0.3860 \\
rte & 0.6787 & \cellcolor{green!40} 0.6077 \\
winogrande & 0.7537 & \cellcolor{green!40} 0.7414 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.498294 | 0.518203 |
| arc_easy      |   0.826178 | 0.807029 |
| boolq         |   0.836086 | 0.813456 |
| hellaswag     |   0.606552 | 0.698981 |
| openbookqa    |   0.32     | 0.387333 |
| rte           |   0.6787   | 0.67509  |
| winogrande    |   0.753749 | 0.743225 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4983 & 0.5182 \\
arc_easy & 0.8262 & 0.8070 \\
boolq & 0.8361 & 0.8135 \\
hellaswag & 0.6066 & 0.6990 \\
openbookqa & 0.3200 & 0.3873 \\
rte & 0.6787 & 0.6751 \\
winogrande & 0.7537 & 0.7432 \\
\bottomrule
\end{tabular}
```
