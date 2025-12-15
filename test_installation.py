#!/usr/bin/env python3
"""Quick test script to verify the installation and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    try:
        import torch
        import networkx as nx
        import matplotlib.pyplot as plt
        
        from social_network_analysis.data.dataset import SocialNetworkDataset
        from social_network_analysis.models.gnn_models import GCN, GraphSAGE, GAT
        from social_network_analysis.train.trainer import Trainer
        from social_network_analysis.eval.evaluator import ModelEvaluator
        from social_network_analysis.utils.device import get_device, set_seed
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    try:
        from social_network_analysis.data.dataset import SocialNetworkDataset
        from social_network_analysis.models.gnn_models import GCN
        from social_network_analysis.utils.device import get_device, set_seed
        
        # Set seed
        set_seed(42)
        
        # Get device
        device = get_device()
        print(f"✅ Device: {device}")
        
        # Create dataset
        dataset = SocialNetworkDataset(seed=42)
        dataset.generate_synthetic_network(n_nodes=50, network_type="barabasi_albert")
        dataset.add_node_features(feature_type="random", n_features=5)
        dataset.add_node_labels(label_type="community", n_classes=3)
        
        data = dataset.to_pyg_data()
        print(f"✅ Dataset created: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
        
        # Create model
        model = GCN(input_dim=5, hidden_dim=32, output_dim=3)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            print(f"✅ Forward pass successful: {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Social Network Analysis - Installation Test")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("🎉 All tests passed! The installation is working correctly.")
        print("\nNext steps:")
        print("- Run training: python scripts/train.py")
        print("- Launch demo: python scripts/run_demo.py")
        print("- Try the notebook: jupyter notebook notebooks/social_network_analysis_demo.ipynb")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
