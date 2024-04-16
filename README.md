# Node Classification on the Cora Dataset

## Overview

This project implements state-of-the-art graph neural network techniques for classifying scientific publications in the Cora dataset. The dataset consists of 2708 machine learning papers categorized into one of seven classes. Each paper is described by a binary word vector indicating the presence of certain words.

The core of this implementation is a Graph Convolutional Network (GCN), optimized through the methodologies discussed in ["Optimization of Graph Neural Networks with Natural Gradient Descent" by Izadi et al. (2020)](https://arxiv.org/pdf/2008.09624v1.pdf).  The original implementation that was adapted to meet the requirements of this project are available at [Russell Izadi's Repository](https://github.com/russellizadi/ssp).

## Methodology

### Graph Convolutional Networks (GCN)

Our model uses GCN layers that capture both node features (word occurrences) and graph topology (citation links) to learn how to classify documents. The GCN's ability to leverage neighborhood information makes it exceptionally well-suited for node classification tasks in citation networks.

### Semi-Supervised Learning

The model is trained in a semi-supervised manner, utilizing labeled and unlabeled nodes together, which allows it to effectively generalize from limited label information spread across the network.

### Optimization

The model is optimized using the Adam optimizer with specific hyperparameters, and further enhanced by techniques such as dropout and L2 regularization to combat overfitting. The training process is also supported by K-Fold cross-validation to ensure the model's robustness and generalizability.

## Getting Started

### Prerequisites

- **Python 3.8+**
- **PyTorch 1.7+**
- **PyTorch Geometric**
- **Anaconda or Miniconda** (Recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ChantelleAA/cora
   cd cora
   ```

2. **Create and Activate a Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate sspedit
   ```

3. **Navigate to the Experiments Directory**
   ```bash
   cd experiments
   ```

4. **Run Experiments**
   ```bash
   ./run.sh
   ```

Ensure that `run.sh` is executable (`chmod +x run.sh`).

### Structure

- **Data Loader**: Loads and preprocesses the Cora dataset.
- **Model Definition**: Defines the GCN model architecture.
- **Training and Evaluation Script**: Manages the training process including parameter updates and assesses model performance on validation sets (using 10-fold cross validation).
- **Utility Scripts**: Provide additional functionalities like logging and data manipulation.

### Outputs

- **Model Checkpoints**: Stored periodically during training.
- **Logs**: Training progress and performance metrics.
- **Predictions**: Classification results on the test set.

## Conclusion

This implementation showcases the effective use of graph convolutional networks for semi-supervised learning on graph-structured data.
