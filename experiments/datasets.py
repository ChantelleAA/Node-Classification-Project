import os
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import json

class CustomCoraLoader:
    """
    Loader for the Cora dataset that initializes paths, loads graph data, and processes it for GNN usage.
    """
    def __init__(self, data_dir, normalize_features=True):
        """
        Constructor for initializing the loader with the dataset directory and normalization option.
        """
        self.data_dir = data_dir
        self.normalize_features = normalize_features
        self.content_file = os.path.join(data_dir, 'cora.content')
        self.cites_file = os.path.join(data_dir, 'cora.cites')
        self.node_id_to_index = {}

    def load_data(self):
        """
        Main method to load, process, and return the graph data as a Data object.
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
        Loads and processes the node features and labels from the content file.
        """
        label_mapping = {
            'Case_Based': 0, 'Genetic_Algorithms': 1, 'Neural_Networks': 2, 
            'Probabilistic_Methods': 3, 'Reinforcement_Learning': 4, 
            'Rule_Learning': 5, 'Theory': 6
        }
        features = []
        labels = []

        with open(self.content_file, 'r') as file:
            for index, line in enumerate(file):
                parts = line.strip().split('\t')
                if len(parts) != (1434 + 2):  # Verify correct format
                    raise ValueError(f"Line {index} in {self.content_file} is malformed.")
                node_id = parts[0]
                self.node_id_to_index[node_id] = index
                features.append([int(x) for x in parts[1:-1]])
                labels.append(label_mapping.get(parts[-1], -1))
                if labels[-1] == -1:
                    raise ValueError(f"Invalid label on line {index} in {self.content_file}.")

        feature_tensor = torch.tensor(features, dtype=torch.float)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return feature_tensor, label_tensor

    def load_cites_file(self):
        """
        Loads and processes the citation links from the cites file.
        """
        edge_list = []

        with open(self.cites_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    raise ValueError(f"Malformed citation line: {line.strip()}")
                source, target = parts
                if source not in self.node_id_to_index or target not in self.node_id_to_index:
                    raise ValueError(f"Citation between non-existent nodes: {source}, {target}")
                source_idx = self.node_id_to_index[source]
                target_idx = self.node_id_to_index[target]
                edge_list.append([source_idx, target_idx])

        if not edge_list:
            raise ValueError("No valid edges found in the .cites file.")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

def save_node_id_to_index(node_id_to_index, filename='node_id_to_index.json'):
    """
    Saves the mapping from node IDs to indices in a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(node_id_to_index, file)

def main():
    """
    Demonstrates loading, processing, and saving the Cora dataset.
    """
    dataset_dir = '../data/'
    data_loader = CustomCoraLoader(data_dir=dataset_dir, normalize_features=True)
    dataset = data_loader.load_data()
    save_node_id_to_index(data_loader.node_id_to_index)
    print(dataset)
    print(f"Number of Nodes: {dataset.x.shape[0]}")
    print(f"Number of Features per Node: {dataset.x.shape[1]}")
    print(f"Number of Edges: {dataset.edge_index.shape[1]}")
    print(f"Number of Classes: {dataset.num_classes}")
    print(f"Node ID to Index Mapping: {data_loader.node_id_to_index}")

if __name__ == "__main__":
    main()
