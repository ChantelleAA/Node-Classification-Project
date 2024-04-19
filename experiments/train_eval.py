from __future__ import division
import time
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import utils as ut
import psgd
from sklearn.model_selection import KFold
import json
import numpy as np
import random

# Set seed for reproducibility in numpy, random, and PyTorch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
# Disable oneDNN optimizations to maintain compatibility across different systems/versions
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the device to GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define paths for saving runs and predictions
path_runs = "runs"
path_predictions = "predictions"

def run(
    dataset, 
    model, 
    str_optimizer, 
    str_preconditioner, 
    runs, 
    epochs, 
    lr, 
    weight_decay, 
    early_stopping,  
    logger, 
    momentum,
    eps,
    update_freq,
    gamma,
    alpha,
    hyperparam
    ):
    """
    Executes the training and evaluation process using cross-validation.
    
    Parameters:
        dataset (Data): Data object containing features, labels, edges.
        model (torch.nn.Module): The GNN model to train.
        str_optimizer (str): Choice of optimizer ('Adam' or 'SGD').
        str_preconditioner (str): Choice of preconditioner ('KFAC' or None).
        runs (int): Number of cross-validation folds.
        epochs (int): Maximum number of epochs per fold.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        early_stopping (int): Early stopping patience; the number of epochs to wait after last improvement.
        logger (str): Path suffix for logging outputs.
        momentum (float): Momentum factor (for momentum-based optimizers).
        eps (float): Epsilon parameter for KFAC.
        update_freq (int): Frequency of updates for the KFAC preconditioner.
        gamma (float): Unused parameter, reserved for future use.
        alpha (float): Moving average parameter for KFAC.
        hyperparam (str): Parameter being tuned, used in naming the log directory.
    """
    if logger:
        # Setup logger for TensorBoard
        logger_name = f"{logger}-{hyperparam}-{eval(hyperparam)}" if hyperparam else logger
        path_logger = os.path.join(path_runs, logger_name)
        ut.empty_dir(path_logger)
        logger_obj = SummaryWriter(log_dir=path_logger)
        print(f"Logging to {path_logger}")

    # Prepare directory for saving predictions
    model_predictions_dir = os.path.join(path_predictions, logger)
    os.makedirs(model_predictions_dir, exist_ok=True)
    predictions_file = os.path.join(model_predictions_dir, 'all_predictions.tsv')

    # Initialize cross-validator
    kf = KFold(n_splits=runs, shuffle=True, random_state=42)
    val_losses, accs, durations = [], [], []

    # Start cross-validation
    with open(predictions_file, "w") as f:
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset.x)):
            # Initialize masks for the current fold
            train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True  

            data = dataset
            data.train_mask = train_mask
            data.val_mask = val_mask

            # Initialize model and optimizer
            model.to(device).reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if str_optimizer == 'Adam' else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            preconditioner = psgd.KFAC(model, eps=eps, update_freq=update_freq, alpha=alpha) if str_preconditioner == 'KFAC' else None

            best_val_loss = float('inf')
            t_start = time.perf_counter()

            # Training loop
            for epoch in range(1, epochs + 1):
                train(model, optimizer, data, preconditioner)
                eval_info = evaluate(model, data)

                # Log performance metrics to TensorBoard
                if logger_obj:
                    for k, v in eval_info.items():
                        logger_obj.add_scalar(f"{k}/Fold_{fold_idx+1}", v, epoch)

                # Update best validation loss
                if eval_info['val loss'] < best_val_loss:
                    best_val_loss = eval_info['val loss']
                    best_val_loss_epoch = epoch

                # Check for early stopping
                if early_stopping and (epoch - best_val_loss_epoch) >= early_stopping:
                    break

            # Save predictions
            node_id_to_index = ut.load_mapping('node_id_to_index.json')
            save_predictions(model, data, f, node_id_to_index)

            # Record fold performance
            t_end = time.perf_counter()
            val_losses.append(best_val_loss)
            accs.append(eval_info['val acc'])
            durations.append(t_end - t_start)

            print(f"Fold {fold_idx + 1}: Best Val Loss: {best_val_loss}")

    if logger_obj:
        logger_obj.close()

    # Report average performance
    print(f'Average Validation Loss: {sum(val_losses) / len(val_losses)}, Average Validation Accuracy: {sum(accs) / len(accs) * 100:.2f}%, Average Duration: {sum(durations) / len(durations):.3f}s')

def train(model, optimizer, data, preconditioner=None, lam=0.):
    """
    Train the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    if preconditioner:
        preconditioner.step()
    optimizer.step()

def evaluate(model, data):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        val_mask = data.val_mask
        loss = F.nll_loss(logits[val_mask], data.y[val_mask]).item()
        pred = logits[val_mask].max(1)[1]
        acc = pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()

    return {'val loss': loss, 'val acc': acc}

def save_predictions(model, data, f, node_id_to_index):
    """
    Save the predictions to a file.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
    label_mapping = {
    'Case_Based': 0,
    'Genetic_Algorithms': 1,
    'Neural_Networks': 2,
    'Probabilistic_Methods': 3,
    'Reinforcement_Learning': 4,
    'Rule_Learning': 5,
    'Theory': 6
                    }    
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    for i, pred in enumerate(preds):
        if data.val_mask[i]:
            paper_id = list(node_id_to_index.keys())[list(node_id_to_index.values()).index(i)]
            class_label = reverse_label_mapping.get(pred.item(), "Unknown")
            f.write(f"{paper_id}\t{class_label}\n")
