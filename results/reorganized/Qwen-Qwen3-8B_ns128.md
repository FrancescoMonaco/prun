# Results for Qwen/Qwen3-8B (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | distribution_matching_no_outliers   | herding    | most_dissimilar   | most_similar   | random     | random_words   | shuffled_zipf   |   unique_tokens | zipf       |
|:--------------|-----------:|:------------|:------------------------|:------------------------------------|:-----------|:------------------|:---------------|:-----------|:---------------|:----------------|----------------:|:-----------|
| arc_challenge |     0.558  | 0.4707      | 0.5574                  | **0.5591**                          | **0.5578** | 0.4655            | 0.4618         | 0.4141     | 0.5558         | 0.5540          |          0.4428 | **0.5574** |
| arc_easy      |     0.8359 | 0.8158      | 0.8166                  | 0.8158                              | 0.8140     | 0.8163            | 0.8163         | 0.8162     | **0.8185**     | **0.8198**      |          0.6449 | **0.8180** |
| boolq         |     0.8661 | 0.7982      | **0.8650**              | 0.8638                              | 0.8624     | 0.8012            | 0.7996         | 0.7531     | 0.8614         | **0.8663**      |          0.703  | **0.8648** |
| hellaswag     |     0.5713 | 0.6513      | 0.6525                  | 0.6524                              | 0.6493     | 0.6503            | 0.6503         | **0.6532** | 0.6517         | **0.6531**      |          0.5253 | **0.6530** |
| openbookqa    |     0.31   | **0.3715**  | 0.3670                  | 0.3665                              | 0.3645     | 0.3655            | 0.3655         | **0.3690** | 0.3679         | 0.3682          |          0.313  | **0.3698** |
| rte           |     0.7834 | 0.7617      | 0.7545                  | 0.7635                              | **0.7762** | **0.7780**        | **0.7780**     | 0.7653     | 0.7541         | 0.7491          |          0.6679 | 0.7563     |
| winogrande    |     0.6772 | 0.6594      | **0.6835**              | **0.6914**                          | 0.6819     | 0.6571            | 0.6562         | 0.6214     | 0.6806         | **0.6867**      |          0.6275 | 0.6817     |
| Mean          |     0.6574 | 0.6469      | 0.6709                  | 0.6732                              | 0.6723     | 0.6477            | 0.6468         | 0.6275     | 0.6700         | 0.6710          |          0.5606 | 0.6716     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_dissimilar & most_similar & random & random_words & shuffled_zipf & unique_tokens & zipf \
\midrule
arc_challenge & 0.5580 & 0.4707 & 0.5574 & \cellcolor{green!40} 0.5591 & \cellcolor{green!25} 0.5578 & 0.4655 & 0.4618 & 0.4141 & 0.5558 & 0.5540 & 0.4428 & \cellcolor{green!10} 0.5574 \"
arc_easy & 0.8359 & 0.8158 & 0.8166 & 0.8158 & 0.8140 & 0.8163 & 0.8163 & 0.8162 & \cellcolor{green!25} 0.8185 & \cellcolor{green!40} 0.8198 & 0.6449 & \cellcolor{green!10} 0.8180 \"
boolq & 0.8661 & 0.7982 & \cellcolor{green!25} 0.8650 & 0.8638 & 0.8624 & 0.8012 & 0.7996 & 0.7531 & 0.8614 & \cellcolor{green!40} 0.8663 & 0.7030 & \cellcolor{green!10} 0.8648 \"
hellaswag & 0.5713 & 0.6513 & 0.6525 & 0.6524 & 0.6493 & 0.6503 & 0.6503 & \cellcolor{green!40} 0.6532 & 0.6517 & \cellcolor{green!25} 0.6531 & 0.5253 & \cellcolor{green!10} 0.6530 \"
openbookqa & 0.3100 & \cellcolor{green!40} 0.3715 & 0.3670 & 0.3665 & 0.3645 & 0.3655 & 0.3655 & \cellcolor{green!10} 0.3690 & 0.3679 & 0.3682 & 0.3130 & \cellcolor{green!25} 0.3698 \"
rte & 0.7834 & 0.7617 & 0.7545 & 0.7635 & \cellcolor{green!10} 0.7762 & \cellcolor{green!40} 0.7780 & \cellcolor{green!25} 0.7780 & 0.7653 & 0.7541 & 0.7491 & 0.6679 & 0.7563 \"
winogrande & 0.6772 & 0.6594 & \cellcolor{green!10} 0.6835 & \cellcolor{green!40} 0.6914 & 0.6819 & 0.6571 & 0.6562 & 0.6214 & 0.6806 & \cellcolor{green!25} 0.6867 & 0.6275 & 0.6817 \"
Mean & 0.6574 & 0.6469 & 0.6709 & 0.6732 & 0.6723 & 0.6477 & 0.6468 & 0.6275 & 0.6700 & 0.6710 & 0.5606 & 0.6716 \"
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.55802  | 0.553043 |                0.557381 |
| arc_easy      |   0.835859 | 0.818673 |                0.816604 |
| boolq         |   0.866055 | 0.862691 |                0.864985 |
| hellaswag     |   0.571301 | 0.649622 |                0.652509 |
| openbookqa    |   0.31     | 0.365333 |                0.367    |
| rte           |   0.783394 | 0.766546 |                0.754513 |
| winogrande    |   0.67719  | 0.677979 |                0.683504 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.5580 & 0.5530 & \cellcolor{blue!15} \textbf{0.5574} \\
arc_easy & 0.8359 & \cellcolor{blue!15} \textbf{0.8187} & 0.8166 \\
boolq & 0.8661 & 0.8627 & \cellcolor{blue!15} \textbf{0.8650} \\
hellaswag & 0.5713 & 0.6496 & \cellcolor{blue!15} \textbf{0.6525} \\
openbookqa & 0.3100 & 0.3653 & \cellcolor{blue!15} \textbf{0.3670} \\
rte & 0.7834 & \cellcolor{blue!15} \textbf{0.7665} & 0.7545 \\
winogrande & 0.6772 & 0.6780 & \cellcolor{blue!15} \textbf{0.6835} \\
\bottomrule
\end{tabular}
```
