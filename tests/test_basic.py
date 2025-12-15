"""Basic tests for social network analysis package."""

import pytest
import torch
import networkx as nx

from social_network_analysis.data.dataset import SocialNetworkDataset
from social_network_analysis.models.gnn_models import GCN, GraphSAGE, GAT
from social_network_analysis.utils.device import get_device, set_seed


def test_dataset_creation():
    """Test dataset creation and basic functionality."""
    dataset = SocialNetworkDataset(seed=42)
    
    # Generate network
    graph = dataset.generate_synthetic_network(n_nodes=100, network_type="barabasi_albert")
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 100
    
    # Add features
    features = dataset.add_node_features(feature_type="random", n_features=10)
    assert features.shape == (100, 10)
    
    # Add labels
    labels = dataset.add_node_labels(label_type="community", n_classes=3)
    assert labels.shape == (100,)
    assert labels.max().item() < 3
    
    # Convert to PyG
    data = dataset.to_pyg_data()
    assert data.x.shape == (100, 10)
    assert data.y.shape == (100,)
    assert data.edge_index.shape[1] == graph.number_of_edges()


def test_model_creation():
    """Test model creation and forward pass."""
    input_dim, hidden_dim, output_dim = 10, 64, 5
    
    # Test GCN
    gcn = GCN(input_dim, hidden_dim, output_dim)
    assert gcn is not None
    
    # Test GraphSAGE
    sage = GraphSAGE(input_dim, hidden_dim, output_dim)
    assert sage is not None
    
    # Test GAT
    gat = GAT(input_dim, hidden_dim, output_dim)
    assert gat is not None


def test_model_forward_pass():
    """Test model forward pass."""
    input_dim, hidden_dim, output_dim = 10, 64, 5
    n_nodes, n_edges = 50, 100
    
    # Create dummy data
    x = torch.randn(n_nodes, input_dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Test GCN
    gcn = GCN(input_dim, hidden_dim, output_dim)
    out = gcn(x, edge_index)
    assert out.shape == (n_nodes, output_dim)
    
    # Test GraphSAGE
    sage = GraphSAGE(input_dim, hidden_dim, output_dim)
    out = sage(x, edge_index)
    assert out.shape == (n_nodes, output_dim)
    
    # Test GAT
    gat = GAT(input_dim, hidden_dim, output_dim)
    out = gat(x, edge_index)
    assert out.shape == (n_nodes, output_dim)


def test_device_management():
    """Test device management utilities."""
    device = get_device()
    assert isinstance(device, torch.device)
    
    # Test seeding
    set_seed(42)
    # Should not raise any exceptions


def test_network_statistics():
    """Test network statistics calculation."""
    dataset = SocialNetworkDataset(seed=42)
    dataset.generate_synthetic_network(n_nodes=50, network_type="barabasi_albert")
    
    stats = dataset.get_network_statistics()
    
    assert "n_nodes" in stats
    assert "n_edges" in stats
    assert "density" in stats
    assert "average_clustering" in stats
    
    assert stats["n_nodes"] == 50
    assert stats["n_edges"] > 0
    assert 0 <= stats["density"] <= 1


def test_train_val_test_split():
    """Test train/validation/test split creation."""
    dataset = SocialNetworkDataset(seed=42)
    dataset.generate_synthetic_network(n_nodes=100, network_type="barabasi_albert")
    dataset.add_node_labels(label_type="community", n_classes=3)
    
    train_mask, val_mask, test_mask = dataset.create_train_val_test_split()
    
    assert train_mask.shape == (100,)
    assert val_mask.shape == (100,)
    assert test_mask.shape == (100,)
    
    # Check that masks are mutually exclusive
    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask & test_mask).any()
    
    # Check that all nodes are assigned
    assert (train_mask | val_mask | test_mask).all()


if __name__ == "__main__":
    pytest.main([__file__])
