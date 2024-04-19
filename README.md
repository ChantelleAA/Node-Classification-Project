# Node Classification on the Cora Dataset

![Cora Dataset image retrieved from Orbifold Consulting](https://github.com/ChantelleAA/Node-Classification-Project/assets/115734837/9e23519d-edbc-420b-bf2e-425078a42bf4)

## Overview

This project implements state-of-the-art graph neural network techniques for classifying scientific publications in the Cora dataset. The dataset consists of 2708 machine learning papers categorized into one of seven classes. Each paper is described by a binary word vector indicating the presence of certain words. More information about the dataset can be found [here](https://github.com/ChantelleAA/Node-Classification-Project/blob/main/data/README).

The core of this implementation is a Graph Convolutional Network (GCN), optimized through the methodologies discussed in ["Optimization of Graph Neural Networks with Natural Gradient Descent" by Izadi et al. (2020)](https://arxiv.org/pdf/2008.09624v1.pdf). The original implementation that was adapted to meet the requirements of this project is available at [Russell Izadi's Repository](https://github.com/russellizadi/ssp).

### Getting Started

#### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ChantelleAA/Node-Classification-Project
   cd Node-Classification-Project
   ```

2. **Create and Activate a Virtual Environment**
   
   To isolate the project dependencies, use a virtual environment. You can use Python's built-in `venv` module:
   ```bash
   python3 -m venv sspe_env
   source sspe_env/bin/activate  # On Windows use `sspe_env\Scripts\activate`
   ```

4. **Install Requirements**

   Install the PyTorch library first.
   ```bash
   pip install torch
   ```
   
   Install all other dependencies using `pip` from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

6. **Navigate to the Experiments Directory**
   ```bash
   cd experiments
   ```

7. **Run Experiments**
   
   Execute the experiment scripts:
   ```bash
   ./run.sh
   ```
   Make sure `run.sh` is executable:
   ```bash
   chmod +x run.sh
   ```

## Methodology

- **Graph Convolutional Networks (GCN)**

This model uses GCN layers that capture both node features (word occurrences) and graph topology (citation links) to learn how to classify documents. The GCN's ability to leverage neighborhood information makes it exceptionally well-suited for node classification tasks in citation networks.

- **Semi-Supervised Learning**

The model is trained in a semi-supervised manner, utilizing labeled and unlabeled nodes together, which allows it to effectively generalize from limited label information spread across the network.

- **Optimization**

The model is optimized using the Adam optimizer with specific hyperparameters, and further enhanced by techniques such as dropout and L2 regularization to combat overfitting. The training process is also supported by K-Fold cross-validation to ensure the model's robustness and generalizability.

## Algorithm

The algorithm implemented for node classification on the Cora dataset via the Graph Convolutional Network (GCN) by going through data processing, model training, and evaluation steps. 

### Step 1: Data Loading and Preprocessing
- **Load Data:** Import the Cora dataset, which includes node features (binary word vectors for each paper) and graph topology (citation links between papers).
- **Preprocess Data:** Convert the raw data into a format suitable for the GCN. This involves creating adjacency matrices to represent graph connections and normalizing these matrices to facilitate efficient gradient flow during model training.

### Step 2: Model Setup
- **Define GCN Architecture:** Configure the multi-layer GCN with specific parameters (e.g., number of layers, hidden units, and activation functions). The model architecture is designed to leverage node features and the graph structure effectively.
- **Initialize Parameters:** Set initial weights and biases, often using strategies like Glorot initialization to ensure that the optimizer has a balanced starting point for parameter updates.

### Step 3: Optimization Configuration
- **Set Optimizer:** Use the Adam optimizer for adjusting model parameters. Adam is chosen for its efficiency in handling sparse gradients and its adaptive learning rate capabilities, which are ideal for the sparse and irregular structure of graph data.
- **Regularization Techniques:** Apply dropout to randomly ignore a subset of features during training, which helps prevent overfitting. Additionally, incorporate L2 regularization to penalize large weights, further enhancing the model's generalization ability.

### Step 4: Training Loop
- **Mini-Batch Processing:** If applicable, divide the data into mini-batches. For GCN, often the entire graph is processed in a single batch due to the interconnected nature of the data.
- **Forward Pass:** Compute node representations by propagating node features through the GCN layers. The layer-wise aggregation of features from neighboring nodes allows the model to learn complex patterns across the graph.
- **Loss Calculation:** Evaluate the prediction error using a suitable loss function, such as cross-entropy for classification tasks.
- **Backward Pass:** Perform backpropagation to compute the gradient of the loss function with respect to each parameter. Update model parameters using the gradients and the Adam optimization algorithm.

### Step 5: Cross-Validation
- **K-Fold Cross-Validation:** Systematically split the dataset into K folds. For each fold, train the model on K-1 folds and validate it on the remaining fold. This process is repeated K times with each fold used exactly once as the validation set.
- **Performance Metrics:** Calculate accuracy, loss, and other relevant metrics for each fold. Aggregate these metrics to assess the overall performance and stability of the model.

### Step 6: Evaluation and Output
- **Test Evaluation:** After model training and validation, assess the model on a separate test set (if available) to evaluate its performance on unseen data.
- **Save Outputs:** Store the model's predictions, performance metrics, and parameter configurations. This includes saving TensorBoard logs for visual analysis of the training process.

### Step 7: Logging and Reproducibility
- **Logging:** Maintain detailed logs of training/validation loss and accuracy, along with system settings and hyperparameters. This is crucial for debugging and refining the model.
- **Fix Random Seeds:** Set seeds for all random number generators used in the project to ensure that results are consistent and reproducible across different runs.
   
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
├── data/                              # Dataset directory
│   ├── cora.content                   # Node feature and label file
│   ├── cora.cites                     # Edge list file
│   └── README.md                      # Description of the data
│
├── experiments/
│   ├── datasets.py                    # Data loading and preprocessing functions
│   ├── gcn.py                         # GCN architecture definition
│   ├── psgd.py                        # Optimization functions
│   ├── train_eval.py                  # Training and evaluation script
│   ├── utils.py                       # Utility functions
│   ├── node_id_to_index.json          # Mapping of node IDs to indices
│   ├── run.sh                         # Shell script to run experiments
│   │
│   ├── runs/                          # TensorBoard logs directory
│   │   └── <model_name>/              # Logs for a specific model
│   │       └── events.out.tfevents.*  # TensorBoard log files
│   │
│   └── predictions/                   # Predictions directory
│       └── <model_name>/              # Predictions for a specific model
│           └── all_predictions.tsv    # Tab-separated predictions file
│
├── requirements.txt                   # Dependencies for code
└── README.md                          # Project documentation
```

## Limitations

- **Scalability**: Current implementation may not efficiently scale to larger graphs or datasets due to memory constraints.
- **Version Dependency**: Specific dependencies on PyTorch, PyTorch Vision, and PyTorch Geometric versions may limit compatibility.
- **Reproducibility**: Although seeds are fixed, slight variations in results may occur due to differences in underlying hardware or software environments.

## Reproducibility

To ensure reproducibility, the seed for all random number generators used in the project is fixed at 42.

## Conclusion

This implementation showcases the effective use of graph convolutional networks for semi-supervised learning on graph-structured data. The setup ensures that any practitioner can replicate the study or extend it to other datasets  or GNN architectures with minimal modifications.
