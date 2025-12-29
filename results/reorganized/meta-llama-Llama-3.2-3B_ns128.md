# Results for meta-llama/Llama-3.2-3B (nsamples=128)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    | least_perplexity   | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|:-------------------|:------------------|:---------------|:-----------|
| arc_challenge |     0.4275 | 0.3311      | **0.4360**              | **0.4384** | **0.3474**         | 0.3327            | 0.3327         | 0.3327     |
| arc_easy      |     0.7449 | **0.7226**  | 0.7176                  | 0.7204     | 0.7151             | **0.7233**        | **0.7233**     | 0.7225     |
| boolq         |     0.7416 | 0.6494      | **0.7376**              | **0.7260** | 0.6466             | **0.6517**        | 0.6517         | 0.6376     |
| hellaswag     |     0.5582 | 0.6354      | **0.6409**              | 0.6364     | **0.6417**         | 0.6355            | 0.6355         | **0.6404** |
| openbookqa    |     0.312  | 0.3400      | **0.3510**              | **0.3450** | 0.3410             | 0.3425            | 0.3425         | **0.3485** |
| rte           |     0.5415 | 0.5740      | 0.5632                  | **0.5848** | 0.5379             | **0.5794**        | **0.5794**     | 0.5722     |
| winogrande    |     0.6938 | 0.6291      | **0.7005**              | **0.7005** | **0.6457**         | 0.6298            | 0.6298         | 0.6321     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & least_perplexity & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.4275 & 0.3311 & \cellcolor{green!25} 0.4360 & \cellcolor{green!40} 0.4384 & \cellcolor{green!10} 0.3474 & 0.3327 & 0.3327 & 0.3327 \\
arc_easy & 0.7449 & \cellcolor{green!10} 0.7226 & 0.7176 & 0.7204 & 0.7151 & \cellcolor{green!40} 0.7233 & \cellcolor{green!25} 0.7233 & 0.7225 \\
boolq & 0.7416 & 0.6494 & \cellcolor{green!40} 0.7376 & \cellcolor{green!25} 0.7260 & 0.6466 & \cellcolor{green!10} 0.6517 & 0.6517 & 0.6376 \\
hellaswag & 0.5582 & 0.6354 & \cellcolor{green!25} 0.6409 & 0.6364 & \cellcolor{green!40} 0.6417 & 0.6355 & 0.6355 & \cellcolor{green!10} 0.6404 \\
openbookqa & 0.3120 & 0.3400 & \cellcolor{green!40} 0.3510 & \cellcolor{green!10} 0.3450 & 0.3410 & 0.3425 & 0.3425 & \cellcolor{green!25} 0.3485 \\
rte & 0.5415 & 0.5740 & 0.5632 & \cellcolor{green!40} 0.5848 & 0.5379 & \cellcolor{green!25} 0.5794 & \cellcolor{green!10} 0.5794 & 0.5722 \\
winogrande & 0.6938 & 0.6291 & \cellcolor{green!40} 0.7005 & \cellcolor{green!25} 0.7005 & \cellcolor{green!10} 0.6457 & 0.6298 & 0.6298 & 0.6321 \\
\bottomrule
\end{tabular}
```
