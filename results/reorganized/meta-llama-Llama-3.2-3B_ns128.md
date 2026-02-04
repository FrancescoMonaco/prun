# Results for meta-llama/Llama-3.2-3B (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | distribution_matching_no_outliers   | herding    | least_perplexity   | most_dissimilar   | most_similar   |   random |   random_words | shuffled_zipf   | unique_tokens   | zipf       |
|:--------------|-----------:|:------------|:------------------------|:------------------------------------|:-----------|:-------------------|:------------------|:---------------|---------:|---------------:|:----------------|:----------------|:-----------|
| arc_challenge |     0.4275 | 0.3311      | 0.4360                  | **0.4411**                          | **0.4384** | 0.3474             | 0.3327            | 0.3327         |   0.3327 |         0.431  | 0.4377          | **0.4396**      | 0.4374     |
| arc_easy      |     0.7449 | **0.7226**  | 0.7176                  | 0.7195                              | 0.7204     | 0.7151             | **0.7233**        | **0.7233**     |   0.7225 |         0.7139 | 0.7165          | 0.7182          | 0.7199     |
| boolq         |     0.7416 | 0.6494      | **0.7376**              | **0.7401**                          | 0.7260     | 0.6466             | 0.6517            | 0.6517         |   0.6376 |         0.7086 | **0.7378**      | 0.7235          | 0.7338     |
| hellaswag     |     0.5582 | 0.6354      | **0.6409**              | 0.6403                              | 0.6364     | **0.6417**         | 0.6355            | 0.6355         |   0.6404 |         0.6352 | 0.6408          | **0.6427**      | 0.6408     |
| openbookqa    |     0.312  | 0.3400      | **0.3510**              | 0.3495                              | 0.3450     | 0.3410             | 0.3425            | 0.3425         |   0.3485 |         0.3449 | 0.3500          | **0.3553**      | **0.3521** |
| rte           |     0.5415 | 0.5740      | 0.5632                  | 0.5776                              | **0.5848** | 0.5379             | **0.5794**        | **0.5794**     |   0.5722 |         0.5451 | 0.5451          | 0.5511          | 0.5578     |
| winogrande    |     0.6938 | 0.6291      | **0.7005**              | 0.6981                              | 0.7005     | 0.6457             | 0.6298            | 0.6298         |   0.6321 |         0.6974 | 0.6997          | **0.7024**      | **0.7041** |
| Mean          |     0.5742 | 0.5545      | 0.5924                  | 0.5952                              | 0.5931     | 0.5536             | 0.5564            | 0.5564         |   0.5551 |         0.5823 | 0.5897          | 0.5904          | 0.5923     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & least_perplexity & most_dissimilar & most_similar & random & random_words & shuffled_zipf & unique_tokens & zipf \
\midrule
arc_challenge & 0.4275 & 0.3311 & 0.4360 & \cellcolor{green!40} 0.4411 & \cellcolor{green!10} 0.4384 & 0.3474 & 0.3327 & 0.3327 & 0.3327 & 0.4310 & 0.4377 & \cellcolor{green!25} 0.4396 & 0.4374 \"
arc_easy & 0.7449 & \cellcolor{green!10} 0.7226 & 0.7176 & 0.7195 & 0.7204 & 0.7151 & \cellcolor{green!40} 0.7233 & \cellcolor{green!25} 0.7233 & 0.7225 & 0.7139 & 0.7165 & 0.7182 & 0.7199 \"
boolq & 0.7416 & 0.6494 & \cellcolor{green!10} 0.7376 & \cellcolor{green!40} 0.7401 & 0.7260 & 0.6466 & 0.6517 & 0.6517 & 0.6376 & 0.7086 & \cellcolor{green!25} 0.7378 & 0.7235 & 0.7338 \"
hellaswag & 0.5582 & 0.6354 & \cellcolor{green!10} 0.6409 & 0.6403 & 0.6364 & \cellcolor{green!25} 0.6417 & 0.6355 & 0.6355 & 0.6404 & 0.6352 & 0.6408 & \cellcolor{green!40} 0.6427 & 0.6408 \"
openbookqa & 0.3120 & 0.3400 & \cellcolor{green!10} 0.3510 & 0.3495 & 0.3450 & 0.3410 & 0.3425 & 0.3425 & 0.3485 & 0.3449 & 0.3500 & \cellcolor{green!40} 0.3553 & \cellcolor{green!25} 0.3521 \"
rte & 0.5415 & 0.5740 & 0.5632 & 0.5776 & \cellcolor{green!40} 0.5848 & 0.5379 & \cellcolor{green!25} 0.5794 & \cellcolor{green!10} 0.5794 & 0.5722 & 0.5451 & 0.5451 & 0.5511 & 0.5578 \"
winogrande & 0.6938 & 0.6291 & \cellcolor{green!10} 0.7005 & 0.6981 & 0.7005 & 0.6457 & 0.6298 & 0.6298 & 0.6321 & 0.6974 & 0.6997 & \cellcolor{green!25} 0.7024 & \cellcolor{green!40} 0.7041 \"
Mean & 0.5742 & 0.5545 & 0.5924 & 0.5952 & 0.5931 & 0.5536 & 0.5564 & 0.5564 & 0.5551 & 0.5823 & 0.5897 & 0.5904 & 0.5923 \"
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.427474 | 0.431456 |                0.436007 |
| arc_easy      |   0.744949 | 0.718575 |                0.717593 |
| boolq         |   0.74159  | 0.735474 |                0.737615 |
| hellaswag     |   0.558156 | 0.64119  |                0.640908 |
| openbookqa    |   0.312    | 0.352667 |                0.351    |
| rte           |   0.541516 | 0.536703 |                0.563177 |
| winogrande    |   0.693765 | 0.702184 |                0.700474 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.4275 & 0.4315 & \cellcolor{blue!15} \textbf{0.4360} \\
arc_easy & 0.7449 & \cellcolor{blue!15} \textbf{0.7186} & 0.7176 \\
boolq & 0.7416 & 0.7355 & \cellcolor{blue!15} \textbf{0.7376} \\
hellaswag & 0.5582 & \cellcolor{blue!15} \textbf{0.6412} & 0.6409 \\
openbookqa & 0.3120 & \cellcolor{blue!15} \textbf{0.3527} & 0.3510 \\
rte & 0.5415 & 0.5367 & \cellcolor{blue!15} \textbf{0.5632} \\
winogrande & 0.6938 & \cellcolor{blue!15} \textbf{0.7022} & 0.7005 \\
\bottomrule
\end{tabular}
```
