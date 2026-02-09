# Results for google/gemma-7b (nsamples=128) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | 0.3778                  | **0.3893**         | 0.3796         | **0.3840** | **0.4009**      |
| arc_easy      |     0.8262 | 0.6750                  | **0.6855**         | 0.6742         | **0.6823** | **0.7001**      |
| boolq         |     0.8361 | 0.6674                  | **0.6745**         | 0.6735         | **0.6910** | **0.6985**      |
| hellaswag     |     0.6066 | 0.4803                  | **0.5010**         | 0.4827         | **0.5010** | **0.5186**      |
| openbookqa    |     0.32   | **0.3107**              | **0.3130**         | 0.3056         | 0.3095     | **0.3185**      |
| rte           |     0.6787 | 0.4788                  | **0.4937**         | **0.4865**     | 0.4801     | **0.4847**      |
| winogrande    |     0.7537 | 0.6367                  | **0.6461**         | 0.6400         | **0.6547** | **0.6565**      |
| Mean          |     0.6457 | 0.5181                  | 0.5290             | 0.5203         | 0.5289     | 0.5397          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.3778 & \cellcolor{green!25} 0.3893 & 0.3796 & \cellcolor{green!10} 0.3840 & \cellcolor{green!40} 0.4009 \\
arc_easy & 0.8262 & 0.6750 & \cellcolor{green!25} 0.6855 & 0.6742 & \cellcolor{green!10} 0.6823 & \cellcolor{green!40} 0.7001 \\
boolq & 0.8361 & 0.6674 & \cellcolor{green!10} 0.6745 & 0.6735 & \cellcolor{green!25} 0.6910 & \cellcolor{green!40} 0.6985 \\
hellaswag & 0.6066 & 0.4803 & \cellcolor{green!25} 0.5010 & 0.4827 & \cellcolor{green!10} 0.5010 & \cellcolor{green!40} 0.5186 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3107 & \cellcolor{green!25} 0.3130 & 0.3056 & 0.3095 & \cellcolor{green!40} 0.3185 \\
rte & 0.6787 & 0.4788 & \cellcolor{green!40} 0.4937 & \cellcolor{green!25} 0.4865 & 0.4801 & \cellcolor{green!10} 0.4847 \\
winogrande & 0.7537 & 0.6367 & \cellcolor{green!10} 0.6461 & 0.6400 & \cellcolor{green!25} 0.6547 & \cellcolor{green!40} 0.6565 \\
Mean & 0.6457 & 0.5181 & 0.5290 & 0.5203 & 0.5289 & 0.5397 \\
\bottomrule
\end{tabular}
```

## Comparison: Distribution Matching vs COLA

| task          |   original |   distribution_matching |
|:--------------|-----------:|------------------------:|
| arc_challenge |   0.498294 |                0.377702 |
| arc_easy      |   0.826178 |                0.674067 |
| boolq         |   0.836086 |                0.664832 |
| hellaswag     |   0.606552 |                0.482482 |
| openbookqa    |   0.32     |                0.309667 |
| rte           |   0.6787   |                0.478339 |
| winogrande    |   0.753749 |                0.636543 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & distribution_matching \\
\midrule
arc_challenge & 0.4983 & 0.3777 \\
arc_easy & 0.8262 & 0.6741 \\
boolq & 0.8361 & 0.6648 \\
hellaswag & 0.6066 & 0.4825 \\
openbookqa & 0.3200 & 0.3097 \\
rte & 0.6787 & 0.4783 \\
winogrande & 0.7537 & 0.6365 \\
\bottomrule
\end{tabular}
```
