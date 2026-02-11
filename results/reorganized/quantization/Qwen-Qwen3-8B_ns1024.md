# Results for Qwen/Qwen3-8B (nsamples=1024) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5405                  | **0.5512**         | 0.5421         | **0.5478** | **0.5427**      |
| arc_easy      |     0.8359 | 0.8035                  | **0.8117**         | 0.8019         | **0.8082** | **0.8043**      |
| boolq         |     0.8661 | **0.8611**              | **0.8612**         | 0.8603         | 0.8586     | **0.8612**      |
| hellaswag     |     0.5713 | **0.6500**              | 0.6492             | 0.6497         | **0.6504** | **0.6500**      |
| openbookqa    |     0.31   | 0.3569                  | **0.3604**         | **0.3584**     | 0.3579     | **0.3590**      |
| rte           |     0.7834 | **0.7712**              | **0.7775**         | 0.7699         | 0.7676     | **0.7762**      |
| winogrande    |     0.6772 | 0.6728                  | **0.6769**         | 0.6711         | **0.6801** | **0.6747**      |
| Mean          |     0.6574 | 0.6651                  | 0.6697             | 0.6647         | 0.6672     | 0.6669          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5405 & \cellcolor{green!40} 0.5512 & 0.5421 & \cellcolor{green!25} 0.5478 & \cellcolor{green!10} 0.5427 \\
arc_easy & 0.8359 & 0.8035 & \cellcolor{green!40} 0.8117 & 0.8019 & \cellcolor{green!25} 0.8082 & \cellcolor{green!10} 0.8043 \\
boolq & 0.8661 & \cellcolor{green!10} 0.8611 & \cellcolor{green!40} 0.8612 & 0.8603 & 0.8586 & \cellcolor{green!25} 0.8612 \\
hellaswag & 0.5713 & \cellcolor{green!10} 0.6500 & 0.6492 & 0.6497 & \cellcolor{green!40} 0.6504 & \cellcolor{green!25} 0.6500 \\
openbookqa & 0.3100 & 0.3569 & \cellcolor{green!40} 0.3604 & \cellcolor{green!10} 0.3584 & 0.3579 & \cellcolor{green!25} 0.3590 \\
rte & 0.7834 & \cellcolor{green!10} 0.7712 & \cellcolor{green!40} 0.7775 & 0.7699 & 0.7676 & \cellcolor{green!25} 0.7762 \\
winogrande & 0.6772 & 0.6728 & \cellcolor{green!25} 0.6769 & 0.6711 & \cellcolor{green!40} 0.6801 & \cellcolor{green!10} 0.6747 \\
Mean & 0.6574 & 0.6651 & 0.6697 & 0.6647 & 0.6672 & 0.6669 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |     cola |   unique_tokens |
|:--------------|-----------:|---------:|----------------:|
| arc_challenge |   0.55802  | 0.524157 |        0.542715 |
| arc_easy      |   0.835859 | 0.791601 |        0.804293 |
| boolq         |   0.866055 | 0.852791 |        0.861162 |
| hellaswag     |   0.571301 | 0.613193 |        0.649958 |
| openbookqa    |   0.31     | 0.345438 |        0.359    |
| rte           |   0.783394 | 0.747969 |        0.776173 |
| winogrande    |   0.67719  | 0.675908 |        0.674724 |
| Mean          |   0.657403 | 0.650151 |        0.666861 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5242 & \cellcolor{blue!15} \textbf{0.5427} \\
arc_easy & 0.8359 & 0.7916 & \cellcolor{blue!15} \textbf{0.8043} \\
boolq & 0.8661 & 0.8528 & \cellcolor{blue!15} \textbf{0.8612} \\
hellaswag & 0.5713 & 0.6132 & \cellcolor{blue!15} \textbf{0.6500} \\
openbookqa & 0.3100 & 0.3454 & \cellcolor{blue!15} \textbf{0.3590} \\
rte & 0.7834 & 0.7480 & \cellcolor{blue!15} \textbf{0.7762} \\
winogrande & 0.6772 & \cellcolor{blue!15} \textbf{0.6759} & 0.6747 \\
Mean & 0.6574 & 0.6502 & 0.6669 \\
\bottomrule
\end{tabular}
```
