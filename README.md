# Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization

## Introduction
Code for the paper "**Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization**".
> Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called *calibration data*) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce **ZipCal**, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method consistently outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on par, in terms of downstream performance, with a state-of-the-art method that relies on model perplexity. The latter becomes prohibitively expensive at large-scale models and datasets, while **ZipCal** is on average $\sim240\times$ faster due to its tractable linear complexity.

## Reproducibility
### Environment Setup
The recreate the environment one could use the `requirements.txt` file with `pip`:

```bash
pip install -r requirements.txt
```

or with `conda` using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will set up all necessary dependencies and tools required for the project.

### Running the experiments
To run the experiments, you can use the provided slurm scripts in the `launch` directory (\*.slurm). Check carefully the parameters for requesting the right resources for the experiments.
For the main experiments we also provide bash scripts that can be used on a local machine (\*.sh).

### Recreating the tables and figures
To recreate the tables and figures one could use the python scripts provided in the `plot` directory. Figures will be saved in the `plots` directory, while tables will be saved in the `results\tables` directory.
