<p align="center">
<picture>
  <img alt="logo" src="docs/assets/logo.png" height="200">
</picture>
<h1 align="center" margin=0px>
ScAPE: Single-cell Analysis of Perturbational Effects
</h1>
</p>

ScAPE is a package implementing the neural network model used in the _Open Problems – Single-Cell Perturbations challenge_ hosted by Kaggle. This is the model we used to generate the submission that achieved top <2% performance (16th position out of 1097 teams) in the [final leaderboard](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/leaderboard).

## Description

In this Kaggle competition, the main objective was to predict the effect of drug perturbations on peripheral blood mononuclear cells (PBMCs) from several patient samples.

Similar to most problems in biological research via omics data, we encountered a high-dimensional feature space (~18k genes) and a low-dimensional observation space (~614 cell/drug combinations) with a low signal-to-noise ratio, where most of the genes show random fluctuations after perturbation. The main data modality to be predicted consisted of signed and log-transformed P-values from differential expression (DE) analysis. In the DE analysis, pseudo-bulk expression profiles from drug-treated cells were compared against the profiles of cells treated with Dimethyl Sulfoxide (DMSO). 

<p align="center">
<picture>
  <img alt="description" src="docs/assets/nn-architecture.png" height="400">
</picture>
<p align="center" margin=0px>
Neural network architecture used for the challenge (ScAPE model).
</p>
</p>


We used a Neural Network that takes as inputs drug and cell features and produces signed log-pvalues. Features were computed as the median of the signed log-pvalues grouped by drugs and cells, calculated from the `de_train.parquet` file (on the training data). Additionally, we also estimated log fold-changes by re-running the LIMMA analysis on the data, to produce a matrix of the same shape as the de_train data but containing log fold changes with estimations of the log fold changes on gene expression. We computed also the median per cell/drug as features for LFCs.

Similar to a Conditional Variational Autoencoder (CVAE), we used cell features both in the encoding part and the decoding part of the NN. Initially, the model consisted of a CVAE that was trained using the cell features as the conditional features to learn an encoding/decoding function conditioned on the particular cell type. However, after testing different ways to train the CVAE (similar to a beta-VAE with different annealing strategies for the Kullback-Leibler divergence term), we finally considered a non probabilistic NN since we did not find any practical advantage or better generalizations in this case with respect to a simpler non-probabilistic NN, much easier to train. 


## Install the package

```
pip install git+https://github.com/scapeML/scape.git
```

## Notebooks

- Basic usage: https://github.com/scapeML/scape/blob/main/docs/notebooks/quick-start.ipynb
- Training pipeline: https://github.com/scapeML/scape/blob/main/docs/notebooks/solution.ipynb
