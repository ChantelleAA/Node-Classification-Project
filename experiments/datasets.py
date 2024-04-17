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
        self.data_dir = data_dir  # Path to the dataset directory
        self.normalize_features = normalize_features  # Flag to determine whether to normalize features
        self.content_file = os.path.join(data_dir, 'cora.content')  # Path to the content file
        self.cites_file = os.path.join(data_dir, 'cora.cites')  # Path to the cites file
        self.node_id_to_index = {}  # Mapping from node ID to index

    def load_data(self):
        """
        Main method to load, process, and return the graph data as a Data object.
        """
        node_features, node_labels = self.load_content_file()  # Load features and labels
        edge_index = self.load_cites_file()  # Load edges
        data = Data(x=node_features, edge_index=edge_index, y=node_labels)  # Create data object
        
        if self.normalize_features:
            NormalizeFeatures()(data)  # Normalize features if flag is set
        
        data.num_classes = int(torch.max(data.y)) + 1  # Determine the number of classes
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
        features = []  # List to hold feature vectors
        labels = []  # List to hold label indices

        with open(self.content_file, 'r') as file:
            for index, line in enumerate(file):
                parts = line.strip().split('\t')
                self.node_id_to_index[parts[0]] = index  # Map node ID to its line index
                features.append([int(x) for x in parts[1:-1]])  # Extract and store features
                labels.append(label_mapping[parts[-1]])  # Map and store labels

        feature_tensor = torch.tensor(features, dtype=torch.float)  # Convert list to tensor
        label_tensor = torch.tensor(labels, dtype=torch.long)  # Convert list to tensor
        return feature_tensor, label_tensor

    def load_cites_file(self):
        """
        Loads and processes the citation links from the cites file to construct the edge index tensor.
        """
        edges = []  # List to hold pairs of indices representing edges

        with open(self.cites_file, 'r') as file:
            for line in file:
                source, target = line.strip().split('\t')
                if source in self.node_id_to_index and target in self.node_id_to_index:
                    source_idx = self.node_id_to_index[source]  # Get index of source node
                    target_idx = self.node_id_to_index[target]  # Get index of target node
                    edges.append([source_idx, target_idx])  # Store edge

        if not edges:
            raise ValueError("No valid edges found. Check the node IDs in the .cites file.")
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Create tensor and transpose
        return edge_index

def save_node_id_to_index(node_id_to_index, filename='node_id_to_index.json'):
    """
    Saves the mapping of node IDs to indices to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(node_id_to_index, file)  # Write dictionary to file in JSON format

def main():
    """
    Main function to demonstrate loading, processing, and using the dataset.
    """
    dataset_dir = '../data/'  # Directory containing the dataset
    data_loader = CustomCoraLoader(data_dir=dataset_dir, normalize_features=True)
    dataset = data_loader.load_data()  # Load and process the dataset
    save_node_id_to_index(data_loader.node_id_to_index)  # Save the node ID to index mapping
    print(dataset)
    print(f"Number of Nodes: {dataset.x.shape[0]}")
    print(f"Number of Features per Node: {dataset.x.shape[1]}")
    print(f"Number of Edges: {dataset.edge_index.shape[1]}")
    print(f"Number of Classes: {dataset.num_classes}")
    print(f"Node ID to Index Mapping: {data_loader.node_id_to_index}")

if __name__ == "__main__":
    main()
