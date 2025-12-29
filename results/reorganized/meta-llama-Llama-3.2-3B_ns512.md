# Results for meta-llama/Llama-3.2-3B (nsamples=512)

## Average across Calibration Groups

| task          |   original | decoupled   | distribution_matching   | herding    | least_perplexity   | most_dissimilar   | most_similar   | random     |
|:--------------|-----------:|:------------|:------------------------|:-----------|:-------------------|:------------------|:---------------|:-----------|
| arc_challenge |     0.4275 | 0.3300      | **0.4403**              | **0.4379** | **0.3468**         | 0.3301            | 0.3301         | 0.3363     |
| arc_easy      |     0.7449 | **0.7238**  | 0.7176                  | 0.7211     | 0.7121             | **0.7225**        | **0.7225**     | 0.7209     |
| boolq         |     0.7416 | 0.6259      | **0.7332**              | **0.7242** | **0.6537**         | 0.6451            | 0.6451         | 0.6375     |
| hellaswag     |     0.5582 | 0.6348      | **0.6414**              | 0.6363     | **0.6424**         | 0.6346            | 0.6346         | **0.6404** |
| openbookqa    |     0.312  | 0.3435      | **0.3490**              | **0.3500** | 0.3460             | 0.3460            | 0.3460         | **0.3490** |
| rte           |     0.5415 | **0.5740**  | **0.5921**              | 0.5668     | **0.5704**         | 0.5686            | 0.5686         | 0.5650     |
| winogrande    |     0.6938 | 0.6286      | **0.7024**              | **0.6981** | **0.6423**         | 0.6281            | 0.6281         | 0.6341     |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrrrr}
\toprule
 & original & decoupled & distribution_matching & herding & least_perplexity & most_dissimilar & most_similar & random \\
\midrule
arc_challenge & 0.4275 & 0.3300 & \cellcolor{green!40} 0.4403 & \cellcolor{green!25} 0.4379 & \cellcolor{green!10} 0.3468 & 0.3301 & 0.3301 & 0.3363 \\
arc_easy & 0.7449 & \cellcolor{green!40} 0.7238 & 0.7176 & 0.7211 & 0.7121 & \cellcolor{green!25} 0.7225 & \cellcolor{green!10} 0.7225 & 0.7209 \\
boolq & 0.7416 & 0.6259 & \cellcolor{green!40} 0.7332 & \cellcolor{green!25} 0.7242 & \cellcolor{green!10} 0.6537 & 0.6451 & 0.6451 & 0.6375 \\
hellaswag & 0.5582 & 0.6348 & \cellcolor{green!25} 0.6414 & 0.6363 & \cellcolor{green!40} 0.6424 & 0.6346 & 0.6346 & \cellcolor{green!10} 0.6404 \\
openbookqa & 0.3120 & 0.3435 & \cellcolor{green!25} 0.3490 & \cellcolor{green!40} 0.3500 & 0.3460 & 0.3460 & 0.3460 & \cellcolor{green!10} 0.3490 \\
rte & 0.5415 & \cellcolor{green!25} 0.5740 & \cellcolor{green!40} 0.5921 & 0.5668 & \cellcolor{green!10} 0.5704 & 0.5686 & 0.5686 & 0.5650 \\
winogrande & 0.6938 & 0.6286 & \cellcolor{green!40} 0.7024 & \cellcolor{green!25} 0.6981 & \cellcolor{green!10} 0.6423 & 0.6281 & 0.6281 & 0.6341 \\
\bottomrule
\end{tabular}
```
