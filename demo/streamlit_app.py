"""Streamlit demo for social network analysis with GNNs."""

import sys
from pathlib import Path

import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from social_network_analysis.data.dataset import SocialNetworkDataset
from social_network_analysis.models.gnn_models import GCN, GraphSAGE, GAT
from social_network_analysis.utils.device import get_device, set_seed


def create_model(model_name, input_dim, hidden_dim, output_dim, **kwargs):
    """Create model based on name and parameters."""
    if model_name == "GCN":
        return GCN(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GraphSAGE":
        return GraphSAGE(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GAT":
        return GAT(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_network(graph, node_colors=None, node_sizes=None, title="Social Network"):
    """Create interactive network visualization."""
    pos = nx.spring_layout(graph, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=node_colors if node_colors is not None else [],
            size=node_sizes if node_sizes is not None else 10,
            colorbar=dict(
                thickness=15,
                title="Centrality",
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2)
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive social network visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#888', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zerolineFalse, showticklabels=False))
                   )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Social Network Analysis with GNNs",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Social Network Analysis with Graph Neural Networks")
    st.markdown("Explore social networks and train GNN models for node classification")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset parameters
    st.sidebar.subheader("Dataset Parameters")
    n_nodes = st.sidebar.slider("Number of nodes", 50, 1000, 200)
    network_type = st.sidebar.selectbox(
        "Network type",
        ["barabasi_albert", "watts_strogatz", "erdos_renyi", "stochastic_block"]
    )
    feature_type = st.sidebar.selectbox(
        "Node features",
        ["random", "degree", "centrality", "community"]
    )
    label_type = st.sidebar.selectbox(
        "Node labels",
        ["community", "random", "degree_based"]
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    model_name = st.sidebar.selectbox("Model", ["GCN", "GraphSAGE", "GAT"])
    hidden_dim = st.sidebar.slider("Hidden dimension", 16, 128, 64)
    num_layers = st.sidebar.slider("Number of layers", 1, 4, 2)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.8, 0.5)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01)
    
    # Generate dataset
    if st.sidebar.button("Generate Dataset"):
        with st.spinner("Generating dataset..."):
            # Set seed for reproducibility
            set_seed(42)
            
            # Create dataset
            dataset = SocialNetworkDataset(seed=42)
            dataset.generate_synthetic_network(
                n_nodes=n_nodes,
                network_type=network_type,
            )
            
            # Add features and labels
            dataset.add_node_features(feature_type=feature_type, n_features=10)
            dataset.add_node_labels(label_type=label_type, n_classes=5)
            
            # Convert to PyG format
            data = dataset.to_pyg_data()
            
            # Create splits
            train_mask, val_mask, test_mask = dataset.create_train_val_test_split()
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            
            # Store in session state
            st.session_state.dataset = dataset
            st.session_state.data = data
            st.session_state.graph = dataset.graph
            
            st.success("Dataset generated successfully!")
    
    # Main content
    if "dataset" in st.session_state:
        dataset = st.session_state.dataset
        data = st.session_state.data
        graph = st.session_state.graph
        
        # Network statistics
        st.subheader("Network Statistics")
        stats = dataset.get_network_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", stats["n_nodes"])
        with col2:
            st.metric("Edges", stats["n_edges"])
        with col3:
            st.metric("Density", f"{stats['density']:.3f}")
        with col4:
            st.metric("Clustering", f"{stats['average_clustering']:.3f}")
        
        # Network visualization
        st.subheader("Network Visualization")
        
        # Calculate centrality for visualization
        centrality = nx.degree_centrality(graph)
        node_colors = [centrality[node] for node in graph.nodes()]
        node_sizes = [centrality[node] * 20 + 5 for node in graph.nodes()]
        
        # Create interactive plot
        fig = visualize_network(graph, node_colors, node_sizes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model training
        st.subheader("Model Training")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Create model
                model = create_model(
                    model_name=model_name,
                    input_dim=data.x.shape[1],
                    hidden_dim=hidden_dim,
                    output_dim=data.y.max().item() + 1,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                
                # Create trainer
                from social_network_analysis.train.trainer import Trainer
                trainer = Trainer(
                    model=model,
                    learning_rate=learning_rate,
                    epochs=epochs,
                )
                
                # Train
                criterion = torch.nn.CrossEntropyLoss()
                history = trainer.train(
                    data=data,
                    train_mask=data.train_mask,
                    val_mask=data.val_mask,
                    criterion=criterion,
                    epochs=epochs,
                    verbose=False,
                )
                
                # Store results
                st.session_state.trainer = trainer
                st.session_state.history = history
                
                st.success("Model trained successfully!")
        
        # Training results
        if "trainer" in st.session_state:
            trainer = st.session_state.trainer
            history = st.session_state.history
            
            st.subheader("Training Results")
            
            # Plot training curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history["train_loss"], label="Train Loss")
            ax1.plot(history["val_loss"], label="Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Validation Loss")
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(history["train_acc"], label="Train Accuracy")
            ax2.plot(history["val_acc"], label="Validation Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Training and Validation Accuracy")
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)
            
            # Test evaluation
            st.subheader("Test Evaluation")
            
            # Load best model
            trainer.load_checkpoint("best_model.pt")
            
            # Evaluate
            from social_network_analysis.eval.evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            
            test_metrics = evaluator.evaluate_node_classification(
                model=trainer.model,
                data=data,
                test_mask=data.test_mask,
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{test_metrics['accuracy']:.3f}")
            with col2:
                st.metric("F1 Macro", f"{test_metrics['f1_macro']:.3f}")
            with col3:
                st.metric("F1 Micro", f"{test_metrics['f1_micro']:.3f}")
            with col4:
                st.metric("Precision Macro", f"{test_metrics['precision_macro']:.3f}")
            
            # Node prediction visualization
            st.subheader("Node Predictions")
            
            # Get predictions
            trainer.model.eval()
            with torch.no_grad():
                out = trainer.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                prob = torch.softmax(out, dim=1)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                "Node": range(data.x.shape[0]),
                "True Label": data.y.numpy(),
                "Predicted Label": pred.numpy(),
                "Confidence": prob.max(dim=1)[0].numpy(),
            })
            
            # Filter for test nodes
            test_pred_df = pred_df[data.test_mask.numpy()]
            
            st.dataframe(test_pred_df.head(20))
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(test_pred_df["True Label"], test_pred_df["Predicted Label"])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix (Test Set)")
            
            st.pyplot(fig)
    
    else:
        st.info("Please generate a dataset first using the sidebar controls.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and PyTorch Geometric")


if __name__ == "__main__":
    main()
