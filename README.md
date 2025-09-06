<p align="center">
  <img alt="ScAPE Logo" src="https://raw.githubusercontent.com/scapeML/scape/main/docs/assets/scape_logo.png" height="200">
</p>

<h1 align="center">ScAPE: Single-cell Analysis of Perturbational Effects</h1>

<p align="center">
  <strong>A baseline ML model to predict cell responses to drug perturbations</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1-o_lT-ttoKS-nbozj2RQusGoi-vm0-XL?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://zenodo.org/records/10617221"><img src="https://img.shields.io/badge/Data-Zenodo-blue.svg" alt="Data"></a>
  <a href="https://github.com/scapeML/scape/blob/main/LICENSE"><img src="https://img.shields.io/github/license/scapeML/scape.svg" alt="License"></a>
</p>

---

## 🏆 Highlights

**Award-winning solution** from the NeurIPS 2023 Single-Cell Perturbations Challenge:
- 🥇 **$10,000 Judges' Prize** for performance and methodology
- 🥈 **2nd place** in post-hoc analysis
- 📊 **Top 2%** overall (16th/1097 teams)

## 🚀 Quick Start

```bash
pip install git+https://github.com/scapeML/scape.git
```

```python
import scape

# data from zenodo can be downloaded via
scape.io.download_from_zenodo(target_dir = ".")

# Train model with drug cross-validation
result = scape.api.train(
    de_file="_data/de_train.parquet",
    lfc_file="_data/lfc_train.parquet", 
    cv_drug="Belinostat",
    n_genes=64
)

# Visualize performance vs baselines
scape.util.plot_result(result._last_train_results)
```

## 📋 Overview

ScAPE is a lightweight neural network (~9.6M parameters for the single-task version) that predicts differential gene expression in response to drug perturbations. Built with **Keras 3** for multi-backend support (TensorFlow, JAX, PyTorch).

### Key Features

- 🎯 **Single or Multi-Task Learning**: Predict p-values only or jointly with fold changes
- 🔄 **Multi-Backend Support**: Choose between TensorFlow, JAX, or PyTorch
- 🎲 **Built-in Ensemble Methods**: Simple blending for robust predictions
- 📊 **Cross-Validation**: Cell-type and drug-based validation strategies
- ⚡ **Efficient**: Handles ~18,000 genes with median-based feature engineering

### Architecture

The model uses median-based feature engineering: for each drug and cell type, we compute median differential expression values across the dataset. This reduces ~18,000 genes to manageable drug/cell signatures while preserving biological signal.
<p align="center">
  <img alt="Architecture" src="docs/assets/nn-architecture.png" width="600">
</p>
Key design choices:

- **Dual conditioning**: Cell features are used in both encoder and decoder (similar to CVAEs)
- **Non-probabilistic**: After testing VAE variants, we found a simpler deterministic NN performed equally well.
- **Multi-source features**: Combines signed log p-values and log fold changes for richer representations


## 💻 Usage

### Basic Training

```bash
# Command line
python -m scape train --n-genes 64 --cv-drug Belinostat _data/de_train.parquet _data/lfc_train.parquet

# Python API
import scape

model = scape.model.create_default_model(
    n_genes=64, 
    df_de=de_data, 
    df_lfc=lfc_data
)

results = model.train(
    val_cells=['NK cells'],
    val_drugs=['Belinostat'],
    epochs=600
)
```

### Multi-Task Learning

Configure the model to jointly predict both p-values and fold changes:

```python
# Multi-task configuration with optimal weights
model.model.compile(
    optimizer=optimizer,
    loss={'slogpval': mrrmse, 'lfc': mrrmse},
    loss_weights={'slogpval': 0.8, 'lfc': 0.2}
)
```

### Backend Selection

```bash
# Use JAX backend (recommended for performance)
KERAS_BACKEND=jax python -m scape train ...

# Use TensorFlow backend
KERAS_BACKEND=tensorflow python -m scape train ...

# Use PyTorch backend
KERAS_BACKEND=torch python -m scape train ...
```

### Ensemble Predictions

Improve robustness with simple ensemble blending:

```python
from sklearn.model_selection import KFold
import numpy as np

# Train multiple models with K-fold
predictions = []
for train_idx, val_idx in KFold(n_splits=5).split(all_combinations):
    model = scape.model.create_default_model(...)
    model.train(...)
    predictions.append(model.predict(test_combinations))

# Blend predictions (median)
ensemble_pred = np.median([p.values for p in predictions], axis=0)
```

### Advanced Configuration

```python
# Custom architecture
config = {
    "encoder_hidden_layer_sizes": [128, 128],
    "decoder_hidden_layer_sizes": [128, 512],
    "outputs": {
        "slogpval": (64, "linear"),
        "lfc": (64, "linear"),  # Multi-task
    },
    "noise": 0.01,
    "dropout": 0.05
}

model = scape.model.create_model(
    n_genes=64,
    df_de=de_data,
    df_lfc=lfc_data,
    config=config
)
```

## 📊 Performance Visualization

<p align="center">
  <img alt="Performance Example" src="docs/assets/example-nk-prednisolone.png" width="600">
</p>

Track model improvement over baselines:
- **Zero baseline**: Always predicts 0 (competition baseline)
- **Median baseline**: Predicts drug-specific medians

## 📚 Resources

- 📓 [Quick Start Tutorial](https://github.com/scapeML/scape/blob/main/docs/notebooks/quick-start.ipynb)
- 📓 [Training Pipeline](https://github.com/scapeML/scape/blob/main/docs/notebooks/solution.ipynb)
- 📓 [Google Colab Demo](https://colab.research.google.com/drive/1-o_lT-ttoKS-nbozj2RQusGoi-vm0-XL?usp=sharing)
- 📄 [Technical Report](https://github.com/scapeML/scape/blob/main/docs/report.pdf)
- 💾 [Dataset (Zenodo)](https://zenodo.org/records/10617221)

## 🛠️ Development

```bash
# Setup with pixi
pixi install
pixi shell -e dev

# Run tests (JAX backend recommended)
KERAS_BACKEND=jax pixi run -e dev test

# Lint & format
pixi run lint
pixi run format
```

## 📖 Citation

```bibtex
@misc{rodriguezmier24scape,
  author = {Rodriguez-Mier, Pablo and Garrido-Rodriguez, Martin},
  title = {ScAPE: Single-cell Analysis of Perturbational Effects},
  year = {2024},
  url = {https://github.com/scapeML/scape}
}
```

