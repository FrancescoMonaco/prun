# Results for google/gemma-7b (nsamples=128)

## Average across Calibration Groups

| task          |   original |   decoupled | distribution_matching   | distribution_matching_no_outliers   | herding    |   most_similar | random     |
|:--------------|-----------:|------------:|:------------------------|:------------------------------------|:-----------|---------------:|:-----------|
| arc_challenge |     0.4983 |      0.3763 | **0.5162**              | **0.5164**                          | **0.5094** |         0.3727 | 0.3829     |
| arc_easy      |     0.8262 |      0.8007 | **0.8050**              | **0.8035**                          | 0.8008     |         0.7996 | **0.8041** |
| boolq         |     0.8361 |      0.6555 | **0.8231**              | **0.8234**                          | **0.8116** |         0.6379 | 0.6567     |
| hellaswag     |     0.6066 |      0.6912 | **0.6962**              | **0.6958**                          | 0.6917     |         0.6908 | **0.6956** |
| openbookqa    |     0.32   |      0.385  | **0.3875**              | **0.3885**                          | 0.3840     |         0.3815 | **0.3862** |
| rte           |     0.6787 |      0.5523 | **0.6137**              | **0.5848**                          | 0.5523     |         0.5542 | **0.6137** |
| winogrande    |     0.7537 |      0.6382 | **0.7411**              | **0.7380**                          | **0.7320** |         0.6347 | 0.6464     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_similar & random \\
\midrule
arc_challenge & 0.4983 & 0.3763 & \cellcolor{green!25} 0.5162 & \cellcolor{green!40} 0.5164 & \cellcolor{green!10} 0.5094 & 0.3727 & 0.3829 \\
arc_easy & 0.8262 & 0.8007 & \cellcolor{green!40} 0.8050 & \cellcolor{green!10} 0.8035 & 0.8008 & 0.7996 & \cellcolor{green!25} 0.8041 \\
boolq & 0.8361 & 0.6555 & \cellcolor{green!25} 0.8231 & \cellcolor{green!40} 0.8234 & \cellcolor{green!10} 0.8116 & 0.6379 & 0.6567 \\
hellaswag & 0.6066 & 0.6912 & \cellcolor{green!40} 0.6962 & \cellcolor{green!25} 0.6958 & 0.6917 & 0.6908 & \cellcolor{green!10} 0.6956 \\
openbookqa & 0.3200 & 0.3850 & \cellcolor{green!25} 0.3875 & \cellcolor{green!40} 0.3885 & 0.3840 & 0.3815 & \cellcolor{green!10} 0.3862 \\
rte & 0.6787 & 0.5523 & \cellcolor{green!40} 0.6137 & \cellcolor{green!10} 0.5848 & 0.5523 & 0.5542 & \cellcolor{green!25} 0.6137 \\
winogrande & 0.7537 & 0.6382 & \cellcolor{green!40} 0.7411 & \cellcolor{green!25} 0.7380 & \cellcolor{green!10} 0.7320 & 0.6347 & 0.6464 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.498294 | 0.516638 |                0.516212 |
| arc_easy      |   0.826178 | 0.807379 |                0.805029 |
| boolq         |   0.836086 | 0.811417 |                0.823089 |
| hellaswag     |   0.606552 | 0.698632 |                0.696201 |
| openbookqa    |   0.32     | 0.386    |                0.3875   |
| rte           |   0.6787   | 0.672684 |                0.613718 |
| winogrande    |   0.753749 | 0.74191  |                0.741121 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.5166} & 0.5162 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.8074} & 0.8050 \\
boolq & 0.8361 & 0.8114 & \cellcolor{blue!15} \textbf{0.8231} \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.6986} & 0.6962 \\
openbookqa & 0.3200 & 0.3860 & \cellcolor{blue!15} \textbf{0.3875} \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.6727} & 0.6137 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.7419} & 0.7411 \\
\bottomrule
\end{tabular}
``` 
