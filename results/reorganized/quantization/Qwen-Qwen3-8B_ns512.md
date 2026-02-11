# Results for Qwen/Qwen3-8B (nsamples=512) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5401                  | **0.5533**         | **0.5430**     | 0.5424     | **0.5439**      |
| arc_easy      |     0.8359 | 0.8002                  | **0.8051**         | 0.8022         | **0.8053** | **0.8054**      |
| boolq         |     0.8661 | 0.8589                  | 0.8593             | **0.8597**     | **0.8609** | **0.8594**      |
| hellaswag     |     0.5713 | 0.6485                  | **0.6489**         | 0.6488         | **0.6501** | **0.6504**      |
| openbookqa    |     0.31   | 0.3611                  | **0.3630**         | **0.3636**     | 0.3568     | **0.3661**      |
| rte           |     0.7834 | **0.7771**              | **0.7802**         | 0.7671         | **0.7739** | 0.7703          |
| winogrande    |     0.6772 | **0.6827**              | **0.6787**         | 0.6771         | 0.6719     | **0.6838**      |
| Mean          |     0.6574 | 0.6669                  | 0.6698             | 0.6659         | 0.6659     | 0.6685          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5401 & \cellcolor{green!40} 0.5533 & \cellcolor{green!10} 0.5430 & 0.5424 & \cellcolor{green!25} 0.5439 \\
arc_easy & 0.8359 & 0.8002 & \cellcolor{green!10} 0.8051 & 0.8022 & \cellcolor{green!25} 0.8053 & \cellcolor{green!40} 0.8054 \\
boolq & 0.8661 & 0.8589 & 0.8593 & \cellcolor{green!25} 0.8597 & \cellcolor{green!40} 0.8609 & \cellcolor{green!10} 0.8594 \\
hellaswag & 0.5713 & 0.6485 & \cellcolor{green!10} 0.6489 & 0.6488 & \cellcolor{green!25} 0.6501 & \cellcolor{green!40} 0.6504 \\
openbookqa & 0.3100 & 0.3611 & \cellcolor{green!10} 0.3630 & \cellcolor{green!25} 0.3636 & 0.3568 & \cellcolor{green!40} 0.3661 \\
rte & 0.7834 & \cellcolor{green!25} 0.7771 & \cellcolor{green!40} 0.7802 & 0.7671 & \cellcolor{green!10} 0.7739 & 0.7703 \\
winogrande & 0.6772 & \cellcolor{green!25} 0.6827 & \cellcolor{green!10} 0.6787 & 0.6771 & 0.6719 & \cellcolor{green!40} 0.6838 \\
Mean & 0.6574 & 0.6669 & 0.6698 & 0.6659 & 0.6659 & 0.6685 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.55802  | 0.524931 |        0.543889 |
| arc_easy      |   0.835859 | 0.789681 |        0.805371 |
| boolq         |   0.866055 | 0.851701 |        0.859404 |
| hellaswag     |   0.571301 | 0.612767 |        0.650412 |
| openbookqa    |   0.31     | 0.34725  |        0.366125 |
| rte           |   0.783394 | 0.749097 |        0.770307 |
| winogrande    |   0.67719  | 0.673441 |        0.6838   |
| Mean          |   0.657403 | 0.649838 |        0.668473 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5249 & \cellcolor{blue!15} \textbf{0.5439} \\
arc_easy & 0.8359 & 0.7897 & \cellcolor{blue!15} \textbf{0.8054} \\
boolq & 0.8661 & 0.8517 & \cellcolor{blue!15} \textbf{0.8594} \\
hellaswag & 0.5713 & 0.6128 & \cellcolor{blue!15} \textbf{0.6504} \\
openbookqa & 0.3100 & 0.3473 & \cellcolor{blue!15} \textbf{0.3661} \\
rte & 0.7834 & 0.7491 & \cellcolor{blue!15} \textbf{0.7703} \\
winogrande & 0.6772 & 0.6734 & \cellcolor{blue!15} \textbf{0.6838} \\
Mean & 0.6574 & 0.6498 & 0.6685 \\
\bottomrule
\end{tabular}
```
