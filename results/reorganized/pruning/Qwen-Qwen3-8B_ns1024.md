# Results for Qwen/Qwen3-8B (nsamples=1024) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.558  | 0.5042                  | **0.5070**         | 0.5015         | **0.5065** | **0.5055**      |
| arc_easy      |     0.8359 | 0.7752                  | **0.7816**         | 0.7770         | **0.7790** | **0.7807**      |
| boolq         |     0.8661 | **0.8439**              | **0.8465**         | 0.8417         | **0.8452** | 0.8438          |
| hellaswag     |     0.5713 | 0.5776                  | **0.5803**         | 0.5767         | **0.5785** | **0.5813**      |
| openbookqa    |     0.31   | **0.3351**              | 0.3329             | 0.3329         | **0.3349** | **0.3335**      |
| rte           |     0.7834 | **0.7211**              | 0.7121             | 0.7175         | **0.7220** | **0.7193**      |
| winogrande    |     0.6772 | **0.6711**              | **0.6740**         | **0.6747**     | 0.6704     | 0.6703          |
| Mean          |     0.6574 | 0.6326                  | 0.6335             | 0.6317         | 0.6338     | 0.6335          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.5580 & 0.5042 & \cellcolor{green!40} 0.5070 & 0.5015 & \cellcolor{green!25} 0.5065 & \cellcolor{green!10} 0.5055 \\
arc_easy & 0.8359 & 0.7752 & \cellcolor{green!40} 0.7816 & 0.7770 & \cellcolor{green!10} 0.7790 & \cellcolor{green!25} 0.7807 \\
boolq & 0.8661 & \cellcolor{green!10} 0.8439 & \cellcolor{green!40} 0.8465 & 0.8417 & \cellcolor{green!25} 0.8452 & 0.8438 \\
hellaswag & 0.5713 & 0.5776 & \cellcolor{green!25} 0.5803 & 0.5767 & \cellcolor{green!10} 0.5785 & \cellcolor{green!40} 0.5813 \\
openbookqa & 0.3100 & \cellcolor{green!40} 0.3351 & 0.3329 & 0.3329 & \cellcolor{green!25} 0.3349 & \cellcolor{green!10} 0.3335 \\
rte & 0.7834 & \cellcolor{green!25} 0.7211 & 0.7121 & 0.7175 & \cellcolor{green!40} 0.7220 & \cellcolor{green!10} 0.7193 \\
winogrande & 0.6772 & \cellcolor{green!10} 0.6711 & \cellcolor{green!25} 0.6740 & \cellcolor{green!40} 0.6747 & 0.6704 & 0.6703 \\
Mean & 0.6574 & 0.6326 & 0.6335 & 0.6317 & 0.6338 & 0.6335 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.55802  |                0.501849 |
| arc_easy      |   0.835859 |                0.775708 |
| boolq         |   0.866055 |                0.844648 |
| hellaswag     |   0.571301 |                0.578321 |
| openbookqa    |   0.31     |                0.335    |
| rte           |   0.783394 |                0.720217 |
| winogrande    |   0.67719  |                0.670481 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.5580 & 0.5018 \\
arc_easy & 0.8359 & 0.7757 \\
boolq & 0.8661 & 0.8446 \\
hellaswag & 0.5713 & 0.5783 \\
openbookqa & 0.3100 & 0.3350 \\
rte & 0.7834 & 0.7202 \\
winogrande & 0.6772 & 0.6705 \\
\bottomrule
\end{tabular}
```
