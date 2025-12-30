# Results for google/gemma-7b (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|:---------------|:-----------|
| arc_challenge |     0.4983 | 0.3763      | **0.5162**              | **0.5094** | 0.3727         | **0.3829** |
| arc_easy      |     0.8262 | 0.8007      | **0.8050**              | **0.8008** | 0.7996         | **0.8041** |
| boolq         |     0.8361 | 0.6555      | **0.8231**              | **0.8116** | 0.6379         | **0.6567** |
| hellaswag     |     0.6066 | 0.6912      | **0.6962**              | **0.6917** | 0.6908         | **0.6956** |
| openbookqa    |     0.32   | **0.3850**  | **0.3875**              | 0.3840     | 0.3815         | **0.3862** |
| rte           |     0.6787 | 0.5523      | **0.6137**              | 0.5523     | **0.5542**     | **0.6137** |
| winogrande    |     0.7537 | 0.6382      | **0.7411**              | **0.7320** | 0.6347         | **0.6464** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & most_similar & random \\
\midrule
arc_challenge & 0.4983 & 0.3763 & \cellcolor{green!40} 0.5162 & \cellcolor{green!25} 0.5094 & 0.3727 & \cellcolor{green!10} 0.3829 \\
arc_easy & 0.8262 & 0.8007 & \cellcolor{green!40} 0.8050 & \cellcolor{green!10} 0.8008 & 0.7996 & \cellcolor{green!25} 0.8041 \\
boolq & 0.8361 & 0.6555 & \cellcolor{green!40} 0.8231 & \cellcolor{green!25} 0.8116 & 0.6379 & \cellcolor{green!10} 0.6567 \\
hellaswag & 0.6066 & 0.6912 & \cellcolor{green!40} 0.6962 & \cellcolor{green!10} 0.6917 & 0.6908 & \cellcolor{green!25} 0.6956 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3850 & \cellcolor{green!40} 0.3875 & 0.3840 & 0.3815 & \cellcolor{green!25} 0.3862 \\
rte & 0.6787 & 0.5523 & \cellcolor{green!40} 0.6137 & 0.5523 & \cellcolor{green!10} 0.5542 & \cellcolor{green!25} 0.6137 \\
winogrande & 0.7537 & 0.6382 & \cellcolor{green!40} 0.7411 & \cellcolor{green!25} 0.7320 & 0.6347 & \cellcolor{green!10} 0.6464 \\
\bottomrule
\end{tabular}
```
