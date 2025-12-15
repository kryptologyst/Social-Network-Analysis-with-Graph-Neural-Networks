"""Legacy script - modernized version available in scripts/train.py."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np

from social_network_analysis.data.dataset import SocialNetworkDataset
from social_network_analysis.models.gnn_models import GCN
from social_network_analysis.train.trainer import Trainer
from social_network_analysis.eval.evaluator import ModelEvaluator
from social_network_analysis.utils.device import get_device, set_seed


def main():
    """Legacy social network analysis script - modernized version."""
    print("=" * 60)
    print("Social Network Analysis with Graph Neural Networks")
    print("=" * 60)
    print("Note: This is the legacy script. For the modern version,")
    print("please use: python scripts/train.py")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create modern dataset
    print("\n1. Creating modern dataset...")
    dataset = SocialNetworkDataset(seed=42)
    
    # Generate scale-free network (Barabási-Albert)
    print("   Generating Barabási-Albert network...")
    dataset.generate_synthetic_network(
        n_nodes=200,  # Smaller for demo
        network_type="barabasi_albert",
        m=2  # 2 new edges per new node
    )
    
    # Add node features
    print("   Adding node features...")
    dataset.add_node_features(feature_type="random", n_features=10)
    
    # Add node labels based on communities
    print("   Adding community-based labels...")
    dataset.add_node_labels(label_type="community", n_classes=5)
    
    # Convert to PyG format
    data = dataset.to_pyg_data()
    
    # Create train/val/test splits
    train_mask, val_mask, test_mask = dataset.create_train_val_test_split()
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"   Dataset created: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Calculate centrality measures (legacy NetworkX approach)
    print("\n2. Calculating centrality measures...")
    graph = dataset.graph
    
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    clustering_coeff = nx.clustering(graph)
    
    # Print top influencers by degree centrality
    top_influencers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("   Top influencers (by degree centrality):")
    for node, score in top_influencers:
        print(f"     Node {node}: {score:.3f}")
    
    # Train GNN model
    print("\n3. Training Graph Neural Network...")
    model = GCN(
        input_dim=data.x.shape[1],
        hidden_dim=64,
        output_dim=data.y.max().item() + 1,
        num_layers=2,
        dropout=0.5
    )
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.01,
        epochs=50  # Fewer epochs for demo
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        criterion=criterion,
        epochs=50,
        verbose=True
    )
    
    # Evaluate model
    print("\n4. Evaluating model...")
    evaluator = ModelEvaluator(device=device)
    
    # Load best model
    trainer.load_checkpoint("best_model.pt")
    
    # Test evaluation
    test_metrics = evaluator.evaluate_node_classification(
        model=trainer.model,
        data=data,
        test_mask=test_mask
    )
    
    print("   Test Results:")
    for metric, value in test_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Network statistics
    print("\n5. Network statistics:")
    stats = dataset.get_network_statistics()
    print(f"   Number of nodes: {stats['n_nodes']}")
    print(f"   Number of edges: {stats['n_edges']}")
    print(f"   Average clustering coefficient: {stats['average_clustering']:.3f}")
    print(f"   Density: {stats['density']:.3f}")
    print(f"   Assortativity: {stats['assortativity']:.3f}")
    
    # Create visualization
    print("\n6. Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Calculate node sizes based on degree centrality
    node_sizes = [1000 * degree_centrality[n] for n in graph.nodes()]
    
    # Create layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw network
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color='skyblue',
        node_size=node_sizes,
        edge_color='gray',
        alpha=0.7,
        font_size=8
    )
    
    plt.title("Social Network with Node Size ~ Degree Centrality\n(Trained with Graph Neural Network)")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("social_network_analysis.png", dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'social_network_analysis.png'")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'social_network_analysis.png' for visualization.")
    print("For interactive analysis, run: streamlit run demo/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
