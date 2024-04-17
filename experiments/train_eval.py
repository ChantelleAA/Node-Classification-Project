from __future__ import division
import time
import os
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils as ut
import psgd
from sklearn.model_selection import KFold
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path_runs = "runs"
path_predictions = "predictions"  # Separate directory for predictions

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
    Executes the training process for the graph neural network model across multiple folds.
    
    Args:
        dataset (Data): Dataset object containing features, labels, and edges.
        model (torch.nn.Module): Graph neural network model to be trained.
        str_optimizer (str): Choice of optimizer ('Adam' or 'SGD').
        str_preconditioner (str): Choice of preconditioner ('KFAC' or None).
        runs (int): Number of folds for cross-validation.
        epochs (int): Number of epochs per fold.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay factor for regularization.
        early_stopping (int): Epoch threshold for early stopping.
        logger (str): Base name for logging directories.
        momentum (float): Momentum factor (for SGD).
        eps (float): Tikhonov regularization parameter.
        update_freq (int): Frequency of updates for the preconditioner.
        gamma (float): Exponent for decay within the training loop (optional).
        alpha (float): Running average parameter for updating stats.
        hyperparam (str): Hyperparameter tuning focus (optional).
    """
    if logger:
        logger_name = f"{logger}-{hyperparam}{eval(hyperparam)}" if hyperparam else logger
        path_logger = os.path.join(path_runs, logger_name)
        print(f"path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger_obj = SummaryWriter(log_dir=path_logger)

    # Create directory for model predictions
    model_predictions_dir = os.path.join(path_predictions, logger)
    os.makedirs(model_predictions_dir, exist_ok=True)
    predictions_file = os.path.join(model_predictions_dir, 'all_predictions.tsv')


    kf = KFold(n_splits=runs, shuffle=True, random_state=42)
    val_losses, accs, durations = [], [], []

    with open(predictions_file, "w") as f:
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset.x)):
            # Set up masks for training and validation
            train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True  

            data = dataset
            data.train_mask = train_mask
            data.val_mask = val_mask

            model.to(device).reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if str_optimizer == 'Adam' \
                        else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            preconditioner = psgd.KFAC(model, eps=eps, update_freq=update_freq, alpha=alpha, sua=False, pi=False, constraint_norm=False) \
                             if str_preconditioner == 'KFAC' else None

            best_val_loss = float('inf')
            t_start = time.perf_counter()

            for epoch in range(1, epochs + 1):
                train(model, optimizer, data, preconditioner)
                eval_info = evaluate(model, data)

                if logger_obj:
                    for k, v in eval_info.items():
                        logger_obj.add_scalar(f"{k}/Fold_{fold_idx+1}", v, epoch)

                if eval_info['val loss'] < best_val_loss:
                    best_val_loss = eval_info['val loss']
                    best_val_loss_epoch = epoch

                if early_stopping and (epoch - best_val_loss_epoch) >= early_stopping:
                    break

            node_id_to_index = ut.load_mapping('node_id_to_index.json')
            save_predictions(model, data, f, node_id_to_index)

            t_end = time.perf_counter()
            val_losses.append(best_val_loss)
            accs.append(eval_info['val acc'])
            durations.append(t_end - t_start)

            print(f"Fold {fold_idx + 1}: Best Val Loss: {best_val_loss}")

    if logger_obj:
        logger_obj.close()

    average_accuracy = sum(accs) / len(accs)
    print(f'Average Validation Loss: {sum(val_losses) / len(val_losses)}, Average Validation Accuracy: {average_accuracy * 100:.2f}%, Average Duration: {sum(durations) / len(durations):.3f}s')

def train(model, optimizer, data, preconditioner=None, lam=0.):
    """
    Train the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False
    
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    
    loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step(lam=lam)
    optimizer.step()

def evaluate(model, data):
    """
    Evaluate the model on training and validation sets.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs

def load_node_id_to_index(filename='node_id_to_index.json'):
    """
    Load the node ID to index mapping from a JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def save_predictions(model, data, f, node_id_to_index):
    """
    Save the model predictions to a file.
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
