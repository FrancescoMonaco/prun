# Results for meta-llama/Llama-3.2-3B (nsamples=512) - quantization

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4275 | 0.4280                  | **0.4318**         | **0.4330**     | 0.4280     | **0.4285**      |
| arc_easy      |     0.7449 | 0.7115                  | **0.7179**         | **0.7143**     | 0.7131     | **0.7197**      |
| boolq         |     0.7416 | **0.7213**              | **0.7286**         | 0.7171         | 0.7186     | **0.7261**      |
| hellaswag     |     0.5582 | 0.6337                  | **0.6338**         | 0.6336         | **0.6344** | **0.6351**      |
| openbookqa    |     0.312  | **0.3544**              | **0.3538**         | **0.3554**     | 0.3534     | 0.3514          |
| rte           |     0.5415 | 0.5203                  | **0.5284**         | **0.5483**     | **0.5334** | 0.5171          |
| winogrande    |     0.6938 | 0.6825                  | **0.6925**         | **0.6880**     | 0.6880     | **0.6910**      |
| Mean          |     0.5742 | 0.5788                  | 0.5838             | 0.5842         | 0.5813     | 0.5813          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4275 & 0.4280 & \cellcolor{green!25} 0.4318 & \cellcolor{green!40} 0.4330 & 0.4280 & \cellcolor{green!10} 0.4285 \\
arc_easy & 0.7449 & 0.7115 & \cellcolor{green!25} 0.7179 & \cellcolor{green!10} 0.7143 & 0.7131 & \cellcolor{green!40} 0.7197 \\
boolq & 0.7416 & \cellcolor{green!10} 0.7213 & \cellcolor{green!40} 0.7286 & 0.7171 & 0.7186 & \cellcolor{green!25} 0.7261 \\
hellaswag & 0.5582 & 0.6337 & \cellcolor{green!10} 0.6338 & 0.6336 & \cellcolor{green!25} 0.6344 & \cellcolor{green!40} 0.6351 \\
openbookqa & 0.3120 & \cellcolor{green!25} 0.3544 & \cellcolor{green!10} 0.3538 & \cellcolor{green!40} 0.3554 & 0.3534 & 0.3514 \\
rte & 0.5415 & 0.5203 & \cellcolor{green!10} 0.5284 & \cellcolor{green!40} 0.5483 & \cellcolor{green!25} 0.5334 & 0.5171 \\
winogrande & 0.6938 & 0.6825 & \cellcolor{green!40} 0.6925 & \cellcolor{green!10} 0.6880 & 0.6880 & \cellcolor{green!25} 0.6910 \\
Mean & 0.5742 & 0.5788 & 0.5838 & 0.5842 & 0.5813 & 0.5813 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.427474 |                0.426408 |
| arc_easy      |   0.744949 |                0.710718 |
| boolq         |   0.74159  |                0.721407 |
| hellaswag     |   0.558156 |                0.633838 |
| openbookqa    |   0.312    |                0.353    |
| rte           |   0.541516 |                0.513237 |
| winogrande    |   0.693765 |                0.683899 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4275 & 0.4264 \\
arc_easy & 0.7449 & 0.7107 \\
boolq & 0.7416 & 0.7214 \\
hellaswag & 0.5582 & 0.6338 \\
openbookqa & 0.3120 & 0.3530 \\
rte & 0.5415 & 0.5132 \\
winogrande & 0.6938 & 0.6839 \\
\bottomrule
\end{tabular}
```
