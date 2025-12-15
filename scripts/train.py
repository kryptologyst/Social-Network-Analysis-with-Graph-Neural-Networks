#!/usr/bin/env python3
"""Main training script for social network analysis with GNNs."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from social_network_analysis.data.dataset import SocialNetworkDataset
from social_network_analysis.models.gnn_models import GCN, GraphSAGE, GAT
from social_network_analysis.train.trainer import Trainer
from social_network_analysis.eval.evaluator import ModelEvaluator
from social_network_analysis.utils.device import get_device, set_seed


def create_model(config):
    """Create model based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        PyTorch model.
    """
    model_config = config.model
    
    if model_config.name == "GCN":
        model = GCN(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            use_batch_norm=model_config.use_batch_norm,
        )
    elif model_config.name == "GraphSAGE":
        model = GraphSAGE(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            aggregator=model_config.aggregator,
        )
    elif model_config.name == "GAT":
        model = GAT(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            use_batch_norm=model_config.use_batch_norm,
        )
    else:
        raise ValueError(f"Unknown model: {model_config.name}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN for social network analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--overrides", nargs="*", help="Override config values")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    
    # Set seed for reproducibility
    set_seed(config.device.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = SocialNetworkDataset(
        data_dir=config.paths.data_dir,
        dataset_name=config.dataset.name,
        seed=config.device.seed,
    )
    
    # Generate synthetic network
    dataset.generate_synthetic_network(
        n_nodes=config.dataset.n_nodes,
        network_type=config.dataset.network_type,
    )
    
    # Add features and labels
    dataset.add_node_features(
        feature_type=config.dataset.feature_type,
        n_features=config.dataset.n_features,
    )
    
    dataset.add_node_labels(
        label_type=config.dataset.label_type,
        n_classes=config.dataset.n_classes,
    )
    
    # Convert to PyG format
    data = dataset.to_pyg_data()
    
    # Create train/val/test splits
    train_mask, val_mask, test_mask = dataset.create_train_val_test_split(
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
    )
    
    # Add masks to data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"Dataset created: {data}")
    print(f"Train nodes: {train_mask.sum()}, Val nodes: {val_mask.sum()}, Test nodes: {test_mask.sum()}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model: {config.model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        patience=config.training.patience,
        save_dir=config.paths.checkpoint_dir,
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        criterion=criterion,
        epochs=config.training.epochs,
        verbose=True,
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(device=device)
    
    # Load best model
    trainer.load_checkpoint("best_model.pt")
    
    # Test evaluation
    test_metrics = evaluator.evaluate_node_classification(
        model=trainer.model,
        data=data,
        test_mask=test_mask,
    )
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print network statistics
    stats = dataset.get_network_statistics()
    print("\nNetwork Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Save results
    results = {
        "test_metrics": test_metrics,
        "network_stats": stats,
        "training_history": history,
        "config": OmegaConf.to_yaml(config),
    }
    
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(results, output_dir / "results.pt")
    print(f"\nResults saved to {output_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
