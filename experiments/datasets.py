import os
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import json

class CustomCoraLoader:
    """
    A loader class for the Cora dataset that initializes the dataset path, 
    loads the graph data from content and citation files, and processes them into a format suitable for GNNs.

    Attributes:
        data_dir (str): Directory path where the Cora dataset is located.
        normalize_features (bool): Flag to normalize node features using Torch Geometric's NormalizeFeatures.
    """
    def __init__(self, data_dir, normalize_features=True):
        """
        Initializes the CustomCoraLoader with the directory of the dataset and the option to normalize features.

        Parameters:
            data_dir (str): The directory where the dataset files are stored.
            normalize_features (bool): If set to True, the node features will be normalized.
        """
        self.data_dir = data_dir
        self.normalize_features = normalize_features
        self.content_file = os.path.join(data_dir, 'cora.content')
        self.cites_file = os.path.join(data_dir, 'cora.cites')
        self.node_id_to_index = {} 

    def load_data(self):
        """
        Loads the dataset, processes it, and returns it in a format suitable for use with graph neural networks.

        Returns:
            Data: A Torch Geometric Data object containing graph data.
        """
        node_features, node_labels = self.load_content_file()
        edge_index = self.load_cites_file()
        data = Data(x=node_features, edge_index=edge_index, y=node_labels)
        if self.normalize_features:
            NormalizeFeatures()(data)
        data.num_classes = int(torch.max(data.y)) + 1
        return data

    def load_content_file(self):
        """
        Loads node features and labels from the content file.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature matrix and labels tensor.
        """
        label_mapping = {
            'Case_Based': 0,
            'Genetic_Algorithms': 1,
            'Neural_Networks': 2,
            'Probabilistic_Methods': 3,
            'Reinforcement_Learning': 4,
            'Rule_Learning': 5,
            'Theory': 6
        }
        features = []
        labels = []

        with open(self.content_file, 'r') as file:
            for index, line in enumerate(file):
                parts = line.strip().split('\t')
                self.node_id_to_index[parts[0]] = index
                features.append([int(x) for x in parts[1:-1]])
                labels.append(label_mapping[parts[-1]])

        feature_tensor = torch.tensor(features, dtype=torch.float)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return feature_tensor, label_tensor

    def load_cites_file(self):
        """
        Loads the citation links from the cites file and constructs the edge index tensor.

        Returns:
            torch.Tensor: The edge index tensor representing the graph connectivity.
        """
        edges = []
        with open(self.cites_file, 'r') as file:
            for line in file:
                source, target = line.strip().split('\t')
                if source in self.node_id_to_index and target in self.node_id_to_index:
                    source_idx = self.node_id_to_index[source]
                    target_idx = self.node_id_to_index[target]
                    edges.append([source_idx, target_idx])

        if not edges:
            raise ValueError("No valid edges found. Check the node IDs in the .cites file.")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

def save_node_id_to_index(node_id_to_index, filename='node_id_to_index.json'):
    """
    Saves the mapping from node ID to index to a JSON file.

    Parameters:
        node_id_to_index (dict): A dictionary mapping node IDs to indices.
        filename (str): The filename for storing the JSON data.
    """
    with open(filename, 'w') as file:
        json.dump(node_id_to_index, file)

def main():
    """
    Main function to load data, process it, and print some basic statistics.
    """
    dataset_dir = '../data/'
    data_loader = CustomCoraLoader(data_dir=dataset_dir, normalize_features=True)
    dataset = data_loader.load_data()
    save_node_id_to_index(data_loader.node_id_to_index)
    print(dataset)
    print(f"Number of Nodes: {dataset.x.shape[0]}")
    print(f"Number of Features per Node: {dataset.x.shape[1]}")
    print(f"Number of Edges: {dataset.edge_index.shape[1]}")
    print(f"num_classes: {dataset.num_classes}")
    print(f"{data_loader.node_id_to_index}")

if __name__ == "__main__":
    main()
