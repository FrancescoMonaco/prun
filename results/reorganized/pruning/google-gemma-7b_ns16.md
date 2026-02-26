# Results for google/gemma-7b (nsamples=16) - pruning

## Average across Calibration Groups

| task          |   original | distribution_matching   | least_perplexity   | most_similar   | random     | unique_tokens   |
|:--------------|-----------:|:------------------------|:-------------------|:---------------|:-----------|:----------------|
| arc_challenge |     0.4983 | **0.3818**              | **0.3898**         | 0.3753         | 0.3712     | **0.4056**      |
| arc_easy      |     0.8262 | **0.6720**              | **0.6862**         | 0.6715         | 0.6697     | **0.7035**      |
| boolq         |     0.8361 | **0.6756**              | **0.6747**         | 0.6428         | 0.6726     | **0.7032**      |
| hellaswag     |     0.6066 | 0.4821                  | **0.5163**         | **0.4822**     | 0.4760     | **0.5310**      |
| openbookqa    |     0.32   | **0.3089**              | **0.3096**         | 0.3049         | 0.3045     | **0.3200**      |
| rte           |     0.6787 | 0.4779                  | **0.4810**         | 0.4707         | **0.4856** | **0.4829**      |
| winogrande    |     0.7537 | 0.6349                  | **0.6579**         | **0.6391**     | 0.6379     | **0.6616**      |
| Mean          |     0.6457 | 0.5190                  | 0.5308             | 0.5123         | 0.5168     | 0.5440          |

## LaTeX Table

Note: Requires `\usepackage[table]{xcolor}` in your LaTeX preamble.

```latex
\begin{tabular}{lrrrrrr}
\toprule
 & original & distribution_matching & least_perplexity & most_similar & random & unique_tokens \\
\midrule
arc_challenge & 0.4983 & \cellcolor{green!10} 0.3818 & \cellcolor{green!25} 0.3898 & 0.3753 & 0.3712 & \cellcolor{green!40} 0.4056 \\
arc_easy & 0.8262 & \cellcolor{green!10} 0.6720 & \cellcolor{green!25} 0.6862 & 0.6715 & 0.6697 & \cellcolor{green!40} 0.7035 \\
boolq & 0.8361 & \cellcolor{green!25} 0.6756 & \cellcolor{green!10} 0.6747 & 0.6428 & 0.6726 & \cellcolor{green!40} 0.7032 \\
hellaswag & 0.6066 & 0.4821 & \cellcolor{green!25} 0.5163 & \cellcolor{green!10} 0.4822 & 0.4760 & \cellcolor{green!40} 0.5310 \\
openbookqa & 0.3200 & \cellcolor{green!10} 0.3089 & \cellcolor{green!25} 0.3096 & 0.3049 & 0.3045 & \cellcolor{green!40} 0.3200 \\
rte & 0.6787 & 0.4779 & \cellcolor{green!10} 0.4810 & 0.4707 & \cellcolor{green!40} 0.4856 & \cellcolor{green!25} 0.4829 \\
winogrande & 0.7537 & 0.6349 & \cellcolor{green!25} 0.6579 & \cellcolor{green!10} 0.6391 & 0.6379 & \cellcolor{green!40} 0.6616 \\
Mean & 0.6457 & 0.5190 & 0.5308 & 0.5123 & 0.5168 & 0.5440 \\
\bottomrule
\end{tabular}
```

## Comparison: Unique Tokens vs COLA

| task          |   original |   unique_tokens |
|:--------------|-----------:|----------------:|
| arc_challenge |   0.498294 |        0.40561  |
| arc_easy      |   0.826178 |        0.703546 |
| boolq         |   0.836086 |        0.703211 |
| hellaswag     |   0.606552 |        0.531026 |
| openbookqa    |   0.32     |        0.32     |
| rte           |   0.6787   |        0.482852 |
| winogrande    |   0.753749 |        0.661602 |
| Mean          |   0.645651 |        0.543978 |

### LaTeX Comparison Table

```latex
\begin{tabular}{lrr}
\toprule
 & original & unique_tokens \\
\midrule
arc_challenge & 0.4983 & 0.4056 \\
arc_easy & 0.8262 & 0.7035 \\
boolq & 0.8361 & 0.7032 \\
hellaswag & 0.6066 & 0.5310 \\
openbookqa & 0.3200 & 0.3200 \\
rte & 0.6787 & 0.4829 \\
winogrande & 0.7537 & 0.6616 \\
Mean & 0.6457 & 0.5440 \\
\bottomrule
\end{tabular}
```
