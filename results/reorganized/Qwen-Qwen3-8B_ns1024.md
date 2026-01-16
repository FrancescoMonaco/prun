# Results for Qwen/Qwen3-8B (nsamples=1024)

## Average across Calibration Groups

| task          |   original |   decoupled | distribution_matching   | distribution_matching_no_outliers   | herding    | most_dissimilar   | most_similar   | random     |   unique_tokens | zipf       |
|:--------------|-----------:|------------:|:------------------------|:------------------------------------|:-----------|:------------------|:---------------|:-----------|----------------:|:-----------|
| arc_challenge |     0.558  |      0.4661 | 0.5529                  | **0.5608**                          | **0.5570** | 0.4632            | 0.4589         | 0.4630     |          0.4418 | **0.5577** |
| arc_easy      |     0.8359 |      0.817  | 0.8150                  | **0.8171**                          | 0.8171     | **0.8189**        | **0.8189**     | 0.8162     |          0.6457 | 0.8143     |
| boolq         |     0.8661 |      0.7846 | **0.8659**              | 0.8633                              | **0.8642** | 0.7816            | 0.7793         | 0.7831     |          0.7024 | **0.8645** |
| hellaswag     |     0.5713 |      0.6498 | **0.6529**              | 0.6527                              | 0.6498     | 0.6505            | 0.6505         | **0.6528** |          0.5262 | **0.6532** |
| openbookqa    |     0.31   |      0.36   | **0.3680**              | 0.3675                              | 0.3640     | 0.3640            | 0.3640         | **0.3685** |          0.3113 | **0.3717** |
| rte           |     0.7834 |      0.7617 | 0.7581                  | **0.7653**                          | 0.7617     | **0.7690**        | **0.7690**     | 0.7545     |          0.6691 | 0.7629     |
| winogrande    |     0.6772 |      0.6525 | **0.6894**              | 0.6827                              | **0.6898** | 0.6545            | 0.6544         | 0.6547     |          0.6283 | **0.6917** |
| Mean          |     0.6574 |      0.6417 | 0.6717                  | 0.6728                              | 0.6720     | 0.6431            | 0.6422         | 0.6418     |          0.5607 | 0.6737     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_dissimilar & most_similar & random & unique_tokens & zipf \
\midrule
arc_challenge & 0.5580 & 0.4661 & 0.5529 & \cellcolor{green!40} 0.5608 & \cellcolor{green!10} 0.5570 & 0.4632 & 0.4589 & 0.4630 & 0.4418 & \cellcolor{green!25} 0.5577 \"
arc_easy & 0.8359 & 0.8170 & 0.8150 & \cellcolor{green!10} 0.8171 & 0.8171 & \cellcolor{green!40} 0.8189 & \cellcolor{green!25} 0.8189 & 0.8162 & 0.6457 & 0.8143 \"
boolq & 0.8661 & 0.7846 & \cellcolor{green!40} 0.8659 & 0.8633 & \cellcolor{green!10} 0.8642 & 0.7816 & 0.7793 & 0.7831 & 0.7024 & \cellcolor{green!25} 0.8645 \"
hellaswag & 0.5713 & 0.6498 & \cellcolor{green!25} 0.6529 & 0.6527 & 0.6498 & 0.6505 & 0.6505 & \cellcolor{green!10} 0.6528 & 0.5262 & \cellcolor{green!40} 0.6532 \"
openbookqa & 0.3100 & 0.3600 & \cellcolor{green!10} 0.3680 & 0.3675 & 0.3640 & 0.3640 & 0.3640 & \cellcolor{green!25} 0.3685 & 0.3113 & \cellcolor{green!40} 0.3717 \"
rte & 0.7834 & 0.7617 & 0.7581 & \cellcolor{green!10} 0.7653 & 0.7617 & \cellcolor{green!40} 0.7690 & \cellcolor{green!25} 0.7690 & 0.7545 & 0.6691 & 0.7629 \"
winogrande & 0.6772 & 0.6525 & \cellcolor{green!10} 0.6894 & 0.6827 & \cellcolor{green!25} 0.6898 & 0.6545 & 0.6544 & 0.6547 & 0.6283 & \cellcolor{green!40} 0.6917 \"
Mean & 0.6574 & 0.6417 & 0.6717 & 0.6728 & 0.6720 & 0.6431 & 0.6422 & 0.6418 & 0.5607 & 0.6737 \"
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.55802  | 0.554892 |                0.552901 |
| arc_easy      |   0.835859 | 0.820497 |                0.815025 |
| boolq         |   0.866055 | 0.862181 |                0.865902 |
| hellaswag     |   0.571301 | 0.649539 |                0.652858 |
| openbookqa    |   0.31     | 0.364667 |                0.368    |
| rte           |   0.783394 | 0.766546 |                0.758123 |
| winogrande    |   0.67719  | 0.684294 |                0.689424 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.5580 & \cellcolor{blue!15} \textbf{0.5549} & 0.5529 \\
arc_easy & 0.8359 & \cellcolor{blue!15} \textbf{0.8205} & 0.8150 \\
boolq & 0.8661 & 0.8622 & \cellcolor{blue!15} \textbf{0.8659} \\
hellaswag & 0.5713 & 0.6495 & \cellcolor{blue!15} \textbf{0.6529} \\
openbookqa & 0.3100 & 0.3647 & \cellcolor{blue!15} \textbf{0.3680} \\
rte & 0.7834 & \cellcolor{blue!15} \textbf{0.7665} & 0.7581 \\
winogrande & 0.6772 & 0.6843 & \cellcolor{blue!15} \textbf{0.6894} \\
\bottomrule
\end{tabular}
```
