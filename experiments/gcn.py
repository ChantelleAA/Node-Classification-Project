import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from datasets import CustomCoraLoader
from train_eval import run

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run GCN on the Cora dataset.")
# Arguments for dataset path, number of runs, epochs, learning rate, etc.
parser.add_argument('--dataset', type=str, required=True, help="Directory containing the dataset")
parser.add_argument('--runs', type=int, default=10, help="Number of runs")
parser.add_argument('--epochs', type=int, default=200, help="Number of epochs per run")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay rate")
parser.add_argument('--early_stopping', type=int, default=0, help="Early stopping criteria")
parser.add_argument('--hidden', type=int, default=16, help="Number of hidden units")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
parser.add_argument('--normalize_features', type=bool, default=True, help="Flag to normalize features")
parser.add_argument('--logger', type=str, default=None, help="Logger for TensorBoard")
parser.add_argument('--optimizer', type=str, default='Adam', help="Type of optimizer")
parser.add_argument('--preconditioner', type=str, default=None, help="Type of preconditioner")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum factor")
parser.add_argument('--eps', type=float, default=0.01, help="Epsilon for numerical stability")
parser.add_argument('--update_freq', type=int, default=50, help="Update frequency for optimizer")
parser.add_argument('--gamma', type=float, default=None, help="Gamma hyperparameter")
parser.add_argument('--alpha', type=float, default=None, help="Alpha hyperparameter")
parser.add_argument('--hyperparam', type=str, default=None, help="Hyperparameter to vary")
args = parser.parse_args()

class Net_orig(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_orig, self).__init__()
        # Initialize the first and second GCN layers with the specified number of features and classes
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        # Reset the weights for both GCN layers to reinitialize them before training
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        # Define the forward propagation logic including activation function and dropout
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        # GCN layer with caching to speed up repeated operations
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        # Reinitialize the convolutional layer's weights
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        # Forward pass through ReLU and dropout
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        # Single GCN layer for classification
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        # Reset the convolutional layer parameters
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        # Compute the log softmax for the output
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        # Aggregate CRD and CLS blocks into a complete model
        self.crd = CRD(dataset.num_features, args.hidden, args.dropout)
        self.cls = CLS(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        # Reset all parameters in CRD and CLS blocks
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        # Define the complete model's forward pass
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x

# Load the dataset and prepare the experiment setup
dataset = CustomCoraLoader(data_dir=args.dataset, normalize_features=args.normalize_features)
kwargs = {
    'dataset': dataset.load_data(), 
    'model': Net(dataset.load_data()), 
    'str_optimizer': args.optimizer, 
    'str_preconditioner': args.preconditioner, 
    'runs': args.runs, 
    'epochs': args.epochs, 
    'lr': args.lr, 
    'weight_decay': args.weight_decay, 
    'early_stopping': args.early_stopping, 
    'logger': args.logger, 
    'momentum': args.momentum,
    'eps': args.eps,
    'update_freq': args.update_freq,
    'gamma': args.gamma,
    'alpha': args.alpha,
    'hyperparam': args.hyperparam
}

# Handle different hyperparameters configurations or a standard run
if args.hyperparam:
    for param in (np.logspace(-3, 0, 10) if args.hyperparam == 'eps' else
                  [4, 8, 16, 32, 64, 128] if args.hyperparam == 'update_freq' else
                  np.linspace(1., 10., 10)):
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        run(**kwargs)
else:
    run(**kwargs)
