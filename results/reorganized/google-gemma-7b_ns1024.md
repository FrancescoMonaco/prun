# Results for google/gemma-7b (nsamples=1024)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | distribution_matching_no_outliers   | herding    |   most_similar | random     |
|:--------------|-----------:|:------------|:------------------------|:------------------------------------|:-----------|---------------:|:-----------|
| arc_challenge |     0.4983 | 0.3726      | **0.5151**              | **0.5169**                          | **0.5083** |         0.3671 | 0.3808     |
| arc_easy      |     0.8262 | **0.8044**  | 0.8030                  | **0.8037**                          | 0.8010     |         0.8019 | **0.8044** |
| boolq         |     0.8361 | 0.6459      | **0.8254**              | **0.8261**                          | **0.8182** |         0.6437 | 0.6578     |
| hellaswag     |     0.6066 | 0.6914      | **0.6960**              | **0.6958**                          | 0.6923     |         0.6909 | **0.6957** |
| openbookqa    |     0.32   | 0.3820      | **0.3890**              | **0.3875**                          | 0.3835     |         0.383  | **0.3853** |
| rte           |     0.6787 | 0.5439      | **0.5993**              | **0.6155**                          | 0.5578     |         0.5439 | **0.6173** |
| winogrande    |     0.7537 | 0.6294      | **0.7431**              | **0.7391**                          | **0.7316** |         0.6233 | 0.6384     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & most_similar & random \\
\midrule
arc_challenge & 0.4983 & 0.3726 & \cellcolor{green!25} 0.5151 & \cellcolor{green!40} 0.5169 & \cellcolor{green!10} 0.5083 & 0.3671 & 0.3808 \\
arc_easy & 0.8262 & \cellcolor{green!40} 0.8044 & 0.8030 & \cellcolor{green!10} 0.8037 & 0.8010 & 0.8019 & \cellcolor{green!25} 0.8044 \\
boolq & 0.8361 & 0.6459 & \cellcolor{green!25} 0.8254 & \cellcolor{green!40} 0.8261 & \cellcolor{green!10} 0.8182 & 0.6437 & 0.6578 \\
hellaswag & 0.6066 & 0.6914 & \cellcolor{green!40} 0.6960 & \cellcolor{green!25} 0.6958 & 0.6923 & 0.6909 & \cellcolor{green!10} 0.6957 \\
openbookqa & 0.3200 & 0.3820 & \cellcolor{green!40} 0.3890 & \cellcolor{green!25} 0.3875 & 0.3835 & 0.3830 & \cellcolor{green!10} 0.3853 \\
rte & 0.6787 & 0.5439 & \cellcolor{green!10} 0.5993 & \cellcolor{green!25} 0.6155 & 0.5578 & 0.5439 & \cellcolor{green!40} 0.6173 \\
winogrande & 0.7537 & 0.6294 & \cellcolor{green!40} 0.7431 & \cellcolor{green!25} 0.7391 & \cellcolor{green!10} 0.7316 & 0.6233 & 0.6384 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.498294 | 0.517065 |                0.515145 |
| arc_easy      |   0.826178 | 0.807379 |                0.80303  |
| boolq         |   0.836086 | 0.811621 |                0.825382 |
| hellaswag     |   0.606552 | 0.698898 |                0.696027 |
| openbookqa    |   0.32     | 0.387333 |                0.389    |
| rte           |   0.6787   | 0.672684 |                0.599278 |
| winogrande    |   0.753749 | 0.743225 |                0.743094 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.4983 & \cellcolor{blue!15} \textbf{0.5171} & 0.5151 \\
arc_easy & 0.8262 & \cellcolor{blue!15} \textbf{0.8074} & 0.8030 \\
boolq & 0.8361 & 0.8116 & \cellcolor{blue!15} \textbf{0.8254} \\
hellaswag & 0.6066 & \cellcolor{blue!15} \textbf{0.6989} & 0.6960 \\
openbookqa & 0.3200 & 0.3873 & \cellcolor{blue!15} \textbf{0.3890} \\
rte & 0.6787 & \cellcolor{blue!15} \textbf{0.6727} & 0.5993 \\
winogrande & 0.7537 & \cellcolor{blue!15} \textbf{0.7432} & 0.7431 \\
\bottomrule
\end{tabular}
```
