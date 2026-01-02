# Results for meta-llama/Llama-3.2-3B (nsamples=1024)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | distribution_matching_no_outliers   | herding    | least_perplexity   | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:------------------------------------|:-----------|:-------------------|:------------------|:---------------|:-----------|
| arc_challenge |     0.4275 | 0.3236      | **0.4398**              | **0.4403**                          | **0.4388** | 0.3433             | 0.3228            | 0.3228         | 0.3287     |
| arc_easy      |     0.7449 | 0.7227      | 0.7196                  | **0.7234**                          | 0.7225     | 0.7132             | **0.7235**        | **0.7235**     | 0.7185     |
| boolq         |     0.7416 | 0.6196      | **0.7313**              | **0.7361**                          | **0.7323** | 0.6262             | 0.6151            | 0.6151         | 0.6243     |
| hellaswag     |     0.5582 | 0.6353      | **0.6407**              | **0.6410**                          | 0.6354     | **0.6427**         | 0.6360            | 0.6360         | 0.6399     |
| openbookqa    |     0.312  | 0.3480      | **0.3510**              | 0.3500                              | 0.3470     | **0.3530**         | 0.3450            | 0.3450         | **0.3505** |
| rte           |     0.5415 | **0.5704**  | 0.5578                  | 0.5632                              | 0.5632     | 0.5560             | **0.5866**        | **0.5866**     | 0.5632     |
| winogrande    |     0.6938 | 0.6200      | **0.7009**              | **0.6997**                          | **0.7009** | 0.6348             | 0.6162            | 0.6162         | 0.6239     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & distribution_matching_no_outliers & herding & least_perplexity & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.4275 & 0.3236 & \cellcolor{green!25} 0.4398 & \cellcolor{green!40} 0.4403 & \cellcolor{green!10} 0.4388 & 0.3433 & 0.3228 & 0.3228 & 0.3287 \\
arc_easy & 0.7449 & 0.7227 & 0.7196 & \cellcolor{green!10} 0.7234 & 0.7225 & 0.7132 & \cellcolor{green!40} 0.7235 & \cellcolor{green!25} 0.7235 & 0.7185 \\
boolq & 0.7416 & 0.6196 & \cellcolor{green!10} 0.7313 & \cellcolor{green!40} 0.7361 & \cellcolor{green!25} 0.7323 & 0.6262 & 0.6151 & 0.6151 & 0.6243 \\
hellaswag & 0.5582 & 0.6353 & \cellcolor{green!10} 0.6407 & \cellcolor{green!25} 0.6410 & 0.6354 & \cellcolor{green!40} 0.6427 & 0.6360 & 0.6360 & 0.6399 \\
openbookqa & 0.3120 & 0.3480 & \cellcolor{green!25} 0.3510 & 0.3500 & 0.3470 & \cellcolor{green!40} 0.3530 & 0.3450 & 0.3450 & \cellcolor{green!10} 0.3505 \\
rte & 0.5415 & \cellcolor{green!10} 0.5704 & 0.5578 & 0.5632 & 0.5632 & 0.5560 & \cellcolor{green!40} 0.5866 & \cellcolor{green!25} 0.5866 & 0.5632 \\
winogrande & 0.6938 & 0.6200 & \cellcolor{green!40} 0.7009 & \cellcolor{green!10} 0.6997 & \cellcolor{green!25} 0.7009 & 0.6348 & 0.6162 & 0.6162 & 0.6239 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |     cola |   distribution_matching |
|:--------------|-----------:|---------:|------------------------:|
| arc_challenge |   0.427474 | 0.43302  |                0.439846 |
| arc_easy      |   0.744949 | 0.7164   |                0.719592 |
| boolq         |   0.74159  | 0.73792  |                0.731346 |
| hellaswag     |   0.558156 | 0.640958 |                0.640684 |
| openbookqa    |   0.312    | 0.353667 |                0.351    |
| rte           |   0.541516 | 0.552347 |                0.557762 |
| winogrande    |   0.693765 | 0.704551 |                0.700868 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrrr}
\toprule
 & original & cola & distribution_matching \\
\midrule
arc_challenge & 0.4275 & 0.4330 & \cellcolor{blue!15} \textbf{0.4398} \\
arc_easy & 0.7449 & 0.7164 & \cellcolor{blue!15} \textbf{0.7196} \\
boolq & 0.7416 & \cellcolor{blue!15} \textbf{0.7379} & 0.7313 \\
hellaswag & 0.5582 & \cellcolor{blue!15} \textbf{0.6410} & 0.6407 \\
openbookqa & 0.3120 & \cellcolor{blue!15} \textbf{0.3537} & 0.3510 \\
rte & 0.5415 & 0.5523 & \cellcolor{blue!15} \textbf{0.5578} \\
winogrande & 0.6938 & \cellcolor{blue!15} \textbf{0.7046} & 0.7009 \\
\bottomrule
\end{tabular}
```
