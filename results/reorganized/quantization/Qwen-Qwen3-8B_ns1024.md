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

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.55802  |                0.538254 |
| arc_easy      |   0.835859 |                0.801066 |
| boolq         |   0.866055 |                0.861315 |
| hellaswag     |   0.571301 |                0.650103 |
| openbookqa    |   0.31     |                0.355333 |
| rte           |   0.783394 |                0.768953 |
| winogrande    |   0.67719  |                0.674559 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.5580 & 0.5383 \\
arc_easy & 0.8359 & 0.8011 \\
boolq & 0.8661 & 0.8613 \\
hellaswag & 0.5713 & 0.6501 \\
openbookqa & 0.3100 & 0.3553 \\
rte & 0.7834 & 0.7690 \\
winogrande & 0.6772 & 0.6746 \\
\bottomrule
\end{tabular}
```
