# Results for Qwen/Qwen3-1.7B (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------|:---------------|:-----------|
| arc_challenge |     0.3985 | **0.2897**  | **0.3029**        | 0.2833         | **0.2952** |
| boolq         |     0.7749 | **0.6554**  | **0.6654**        | **0.6547**     | 0.6489     |
| winogrande    |     0.6109 | 0.5604      | **0.5675**        | **0.5793**     | **0.5659** |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrr}
\toprule
 & original & decoupled & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.3985 & \cellcolor{green!10} 0.2897 & \cellcolor{green!40} 0.3029 & 0.2833 & \cellcolor{green!25} 0.2952 \\
boolq & 0.7749 & \cellcolor{green!25} 0.6554 & \cellcolor{green!40} 0.6654 & \cellcolor{green!10} 0.6547 & 0.6489 \\
winogrande & 0.6109 & 0.5604 & \cellcolor{green!25} 0.5675 & \cellcolor{green!40} 0.5793 & \cellcolor{green!10} 0.5659 \\
\bottomrule
\end{tabular}
```

