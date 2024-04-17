# Node Classification on the Cora Dataset

![Cora Dataset image retrieved from Orbifold Consulting](https://github.com/ChantelleAA/Node-Classification-Project/assets/115734837/9e23519d-edbc-420b-bf2e-425078a42bf4)

## Overview

This project implements state-of-the-art graph neural network techniques for classifying scientific publications in the Cora dataset. The dataset consists of 2708 machine learning papers categorized into one of seven classes. Each paper is described by a binary word vector indicating the presence of certain words.

The core of this implementation is a Graph Convolutional Network (GCN), optimized through the methodologies discussed in ["Optimization of Graph Neural Networks with Natural Gradient Descent" by Izadi et al. (2020)](https://arxiv.org/pdf/2008.09624v1.pdf). The original implementation that was adapted to meet the requirements of this project is available at [Russell Izadi's Repository](https://github.com/russellizadi/ssp).

## Methodology

### Graph Convolutional Networks (GCN)

Our model uses GCN layers that capture both node features (word occurrences) and graph topology (citation links) to learn how to classify documents. The GCN's ability to leverage neighborhood information makes it exceptionally well-suited for node classification tasks in citation networks.

### Semi-Supervised Learning

The model is trained in a semi-supervised manner, utilizing labeled and unlabeled nodes together, which allows it to effectively generalize from limited label information spread across the network.

### Optimization

The model is optimized using the Adam optimizer with specific hyperparameters, and further enhanced by techniques such as dropout and L2 regularization to combat overfitting. The training process is also supported by K-Fold cross-validation to ensure the model's robustness and generalizability.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **PyTorch 1.7+**
- **PyTorch Geometric**
- **Anaconda or Miniconda** (Recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ChantelleAA/Node-Classification-Project
   cd Node-Classification-Project
   ```

2. **Create and Activate a Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate sspe
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
- **Optimization**: Defines optimization methods used.
- **Training and Evaluation Script**: Manages the training process including parameter updates and assesses model performance on validation sets (using 10-fold cross-validation).
- **Utility Scripts**: Provide additional functionalities like logging and data manipulation.

### Outputs

- **Model Checkpoints**: Stored periodically during training.
- **Logs**: Training progress and performance metrics.
- **Predictions**: Classification results on the test set.

## Results and File Structure

### Results

The outcomes of this project include:
- **Accuracy and Loss Metrics**: Documented through TensorBoard logs and output console.
- **Predictions**: Predictions of classes for each paper recorded in .tsv file.

### File Structure

The initial file structure contains all directories as shown below except the runs and predictions directories which are added and updated after the code has been run.

```
Node-Classification-Project/
│
├── data/                       # Dataset directory
│   ├── cora.content            # Node feature and label file
│   ├── cora.cites              # Edge list file
│   └── README.md               # Data description
│
├── runs/                       # TensorBoard logs directory
│   └── experiment-1/           # Specific experiment logs
│       └── events.out.tfevents.*
│
├── predictions/                # Predictions directory
│   └── experiment-1/           # Predictions for specific experiment
│       └── all_predictions.tsv
│
├── experiments/
|   ├── datasets.py             # Data loading and preprocessing functions
|   ├── gcn.py                  # GCN architecture definition
|   ├── psgd.py                 # Optimization functions
|   ├── train_eval.py           # Training and evaluation script
|   ├── utils.py                # Utility functions
|   ├── node_id_to_index.json   # Index list for paper ids
│   └── run.sh                  # executable file for project
|
├── environment.yml             # Environment definition script
└── README.md                   # This file

```

## Limitations

- **Scalability**: Current implementation may not efficiently scale to larger graphs or datasets due to memory constraints.
- **Version Dependency**: Specific dependencies on PyTorch and PyTorch-vision

orch Geometric versions may limit compatibility.
- **Reproducibility**: Although seeds are fixed, slight variations in results may occur due to differences in underlying hardware or software environments.

## Reproducibility

To ensure reproducibility, the seed for all random number generators used in the project is fixed at 42.

## Conclusion

This implementation showcases the effective use of graph convolutional networks for semi-supervised learning on graph-structured data. The setup ensures that any practitioner can replicate the study or extend it to other datasets or GNN architectures.
