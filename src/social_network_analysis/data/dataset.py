"""Data loading and preprocessing utilities for social network analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx, to_networkx


class SocialNetworkDataset:
    """Dataset class for social network analysis with GNN support."""
    
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        dataset_name: str = "synthetic",
        seed: int = 42,
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory to store/load data.
            dataset_name: Name of the dataset to load.
            seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.seed = seed
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.graph: Optional[nx.Graph] = None
        self.data: Optional[Data] = None
        self.node_features: Optional[torch.Tensor] = None
        self.node_labels: Optional[torch.Tensor] = None
        self.edge_features: Optional[torch.Tensor] = None
        
    def generate_synthetic_network(
        self,
        n_nodes: int = 1000,
        network_type: str = "barabasi_albert",
        **kwargs,
    ) -> nx.Graph:
        """Generate a synthetic social network.
        
        Args:
            n_nodes: Number of nodes in the network.
            network_type: Type of network to generate.
            **kwargs: Additional parameters for network generation.
            
        Returns:
            nx.Graph: Generated network.
        """
        if network_type == "barabasi_albert":
            m = kwargs.get("m", 3)  # Number of edges to attach from a new node
            self.graph = nx.barabasi_albert_graph(n_nodes, m, seed=self.seed)
        elif network_type == "watts_strogatz":
            k = kwargs.get("k", 4)  # Each node is connected to k nearest neighbors
            p = kwargs.get("p", 0.1)  # Probability of rewiring
            self.graph = nx.watts_strogatz_graph(n_nodes, k, p, seed=self.seed)
        elif network_type == "erdos_renyi":
            p = kwargs.get("p", 0.01)  # Probability of edge creation
            self.graph = nx.erdos_renyi_graph(n_nodes, p, seed=self.seed)
        elif network_type == "stochastic_block":
            n_communities = kwargs.get("n_communities", 5)
            community_sizes = [n_nodes // n_communities] * n_communities
            p_in = kwargs.get("p_in", 0.1)  # Intra-community edge probability
            p_out = kwargs.get("p_out", 0.01)  # Inter-community edge probability
            self.graph = nx.stochastic_block_model(
                community_sizes, [[p_in, p_out], [p_out, p_in]], seed=self.seed
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        return self.graph
    
    def add_node_features(
        self,
        feature_type: str = "random",
        n_features: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """Add node features to the graph.
        
        Args:
            feature_type: Type of features to generate.
            n_features: Number of features per node.
            **kwargs: Additional parameters for feature generation.
            
        Returns:
            torch.Tensor: Node features matrix.
        """
        if self.graph is None:
            raise ValueError("Graph must be generated first")
        
        n_nodes = self.graph.number_of_nodes()
        
        if feature_type == "random":
            self.node_features = torch.randn(n_nodes, n_features)
        elif feature_type == "degree":
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            self.node_features = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
        elif feature_type == "centrality":
            centrality_type = kwargs.get("centrality_type", "degree")
            if centrality_type == "degree":
                centrality = nx.degree_centrality(self.graph)
            elif centrality_type == "betweenness":
                centrality = nx.betweenness_centrality(self.graph)
            elif centrality_type == "closeness":
                centrality = nx.closeness_centrality(self.graph)
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")
            
            centrality_values = [centrality[n] for n in self.graph.nodes()]
            self.node_features = torch.tensor(centrality_values, dtype=torch.float).unsqueeze(1)
        elif feature_type == "community":
            # Use community detection to create features
            communities = nx.community.greedy_modularity_communities(self.graph)
            n_communities = len(communities)
            features = torch.zeros(n_nodes, n_communities)
            for i, community in enumerate(communities):
                for node in community:
                    features[node, i] = 1.0
            self.node_features = features
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        return self.node_features
    
    def add_node_labels(
        self,
        label_type: str = "community",
        n_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """Add node labels for classification tasks.
        
        Args:
            label_type: Type of labels to generate.
            n_classes: Number of classes (if None, will be inferred).
            
        Returns:
            torch.Tensor: Node labels.
        """
        if self.graph is None:
            raise ValueError("Graph must be generated first")
        
        n_nodes = self.graph.number_of_nodes()
        
        if label_type == "community":
            communities = nx.community.greedy_modularity_communities(self.graph)
            labels = torch.zeros(n_nodes, dtype=torch.long)
            for i, community in enumerate(communities):
                for node in community:
                    labels[node] = i
            self.node_labels = labels
        elif label_type == "random":
            if n_classes is None:
                n_classes = 5
            self.node_labels = torch.randint(0, n_classes, (n_nodes,))
        elif label_type == "degree_based":
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            # Create labels based on degree quartiles
            degree_array = np.array(degrees)
            quartiles = np.percentile(degree_array, [25, 50, 75])
            labels = torch.zeros(n_nodes, dtype=torch.long)
            for i, degree in enumerate(degrees):
                if degree <= quartiles[0]:
                    labels[i] = 0
                elif degree <= quartiles[1]:
                    labels[i] = 1
                elif degree <= quartiles[2]:
                    labels[i] = 2
                else:
                    labels[i] = 3
            self.node_labels = labels
        else:
            raise ValueError(f"Unknown label type: {label_type}")
        
        return self.node_labels
    
    def add_edge_features(
        self,
        feature_type: str = "random",
        n_features: int = 5,
    ) -> torch.Tensor:
        """Add edge features to the graph.
        
        Args:
            feature_type: Type of features to generate.
            n_features: Number of features per edge.
            
        Returns:
            torch.Tensor: Edge features matrix.
        """
        if self.graph is None:
            raise ValueError("Graph must be generated first")
        
        n_edges = self.graph.number_of_edges()
        
        if feature_type == "random":
            self.edge_features = torch.randn(n_edges, n_features)
        elif feature_type == "weight":
            # Use edge weights as features
            weights = [self.graph[u][v].get("weight", 1.0) for u, v in self.graph.edges()]
            self.edge_features = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        elif feature_type == "distance":
            # Use shortest path distance as features
            distances = []
            for u, v in self.graph.edges():
                try:
                    dist = nx.shortest_path_length(self.graph, u, v)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    distances.append(0)
            self.edge_features = torch.tensor(distances, dtype=torch.float).unsqueeze(1)
        else:
            raise ValueError(f"Unknown edge feature type: {feature_type}")
        
        return self.edge_features
    
    def to_pyg_data(self) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object.
        
        Returns:
            Data: PyTorch Geometric data object.
        """
        if self.graph is None:
            raise ValueError("Graph must be generated first")
        
        # Convert to PyG format
        self.data = from_networkx(self.graph)
        
        # Add node features if available
        if self.node_features is not None:
            self.data.x = self.node_features
        
        # Add node labels if available
        if self.node_labels is not None:
            self.data.y = self.node_labels
        
        # Add edge features if available
        if self.edge_features is not None:
            self.data.edge_attr = self.edge_features
        
        return self.data
    
    def create_train_val_test_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits for node classification.
        
        Args:
            train_ratio: Ratio of nodes for training.
            val_ratio: Ratio of nodes for validation.
            test_ratio: Ratio of nodes for testing.
            stratify: Whether to stratify by class labels.
            
        Returns:
            Tuple of train, validation, and test masks.
        """
        if self.node_labels is None:
            raise ValueError("Node labels must be generated first")
        
        n_nodes = len(self.node_labels)
        indices = torch.randperm(n_nodes)
        
        if stratify and self.node_labels is not None:
            # Stratified split
            from sklearn.model_selection import train_test_split
            
            train_idx, temp_idx = train_test_split(
                indices.numpy(),
                test_size=1 - train_ratio,
                stratify=self.node_labels.numpy(),
                random_state=self.seed,
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=self.node_labels[temp_idx].numpy(),
                random_state=self.seed,
            )
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        else:
            # Random split
            train_size = int(train_ratio * n_nodes)
            val_size = int(val_ratio * n_nodes)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask
    
    def save_dataset(self, filename: Optional[str] = None) -> None:
        """Save the dataset to disk.
        
        Args:
            filename: Optional custom filename.
        """
        if filename is None:
            filename = f"{self.dataset_name}.pt"
        
        filepath = self.data_dir / filename
        
        # Save PyG data object
        if self.data is not None:
            torch.save(self.data, filepath)
            print(f"Dataset saved to {filepath}")
        else:
            print("No data to save")
    
    def load_dataset(self, filename: Optional[str] = None) -> Data:
        """Load dataset from disk.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Data: Loaded PyTorch Geometric data object.
        """
        if filename is None:
            filename = f"{self.dataset_name}.pt"
        
        filepath = self.data_dir / filename
        
        if filepath.exists():
            self.data = torch.load(filepath)
            print(f"Dataset loaded from {filepath}")
            return self.data
        else:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    def get_network_statistics(self) -> Dict[str, Union[int, float]]:
        """Get basic network statistics.
        
        Returns:
            Dict containing network statistics.
        """
        if self.graph is None:
            raise ValueError("Graph must be generated first")
        
        stats = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_shortest_path_length": nx.average_shortest_path_length(self.graph),
            "diameter": nx.diameter(self.graph),
            "assortativity": nx.degree_assortativity_coefficient(self.graph),
        }
        
        return stats
