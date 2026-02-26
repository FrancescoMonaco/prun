# Results for google/gemma-7b (nsamples=128) - quantization

## Average across Calibration Groups

| task          |   original | random     |
|:--------------|-----------:|:-----------|
| arc_challenge |     0.4983 | **0.4919** |
| arc_easy      |     0.8262 | **0.7693** |
| boolq         |     0.8361 | **0.7713** |
| hellaswag     |     0.6066 | **0.6756** |
| openbookqa    |     0.32   | **0.3685** |
| rte           |     0.6787 | **0.6191** |
| winogrande    |     0.7537 | **0.7269** |
| Mean          |     0.6457 | 0.6318     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrr}
\toprule
 & original & random \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!40} 0.4919 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.7693 \\
boolq & 0.8361 & \cellcolor{green!40} 0.7713 \\
hellaswag & 0.6066 & \cellcolor{green!40} 0.6756 \\
openbookqa & 0.3200 & \cellcolor{green!40} 0.3685 \\
rte & 0.6787 & \cellcolor{green!40} 0.6191 \\
winogrande & 0.7537 & \cellcolor{green!40} 0.7269 \\
Mean & 0.6457 & 0.6318 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |
|:--------------|-----------:|---------:|
| arc_challenge |   0.498294 | 0.516532 |
| arc_easy      |   0.826178 | 0.807134 |
| boolq         |   0.836086 | 0.812997 |
| hellaswag     |   0.606552 | 0.698616 |
| openbookqa    |   0.32     | 0.38475  |
| rte           |   0.6787   | 0.67509  |
| winogrande    |   0.753749 | 0.742699 |
| Mean          |   0.645651 | 0.662545 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & cola \\
\midrule
arc_challenge & 0.4983 & 0.5165 \\
arc_easy & 0.8262 & 0.8071 \\
boolq & 0.8361 & 0.8130 \\
hellaswag & 0.6066 & 0.6986 \\
openbookqa & 0.3200 & 0.3848 \\
rte & 0.6787 & 0.6751 \\
winogrande & 0.7537 & 0.7427 \\
Mean & 0.6457 & 0.6625 \\
\bottomrule
\end{tabular}
```
