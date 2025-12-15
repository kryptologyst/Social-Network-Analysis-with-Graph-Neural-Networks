"""Evaluation utilities for social network analysis models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
import networkx as nx
from torch_geometric.utils import to_networkx


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    @staticmethod
    def node_classification_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Calculate node classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metrics.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {
            "accuracy": accuracy_score(y_true_np, y_pred_np),
            "f1_macro": f1_score(y_true_np, y_pred_np, average="macro"),
            "f1_micro": f1_score(y_true_np, y_pred_np, average="micro"),
            "f1_weighted": f1_score(y_true_np, y_pred_np, average="weighted"),
            "precision_macro": precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
            "precision_micro": precision_score(y_true_np, y_pred_np, average="micro", zero_division=0),
            "recall_macro": recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true_np, y_pred_np, average="micro", zero_division=0),
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy()
            if len(np.unique(y_true_np)) > 2:
                # Multi-class AUC
                metrics["auc_ovr"] = roc_auc_score(y_true_np, y_prob_np, multi_class="ovr")
                metrics["auc_ovo"] = roc_auc_score(y_true_np, y_prob_np, multi_class="ovo")
            else:
                # Binary AUC
                metrics["auc"] = roc_auc_score(y_true_np, y_prob_np[:, 1])
        
        return metrics
    
    @staticmethod
    def link_prediction_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Calculate link prediction metrics.
        
        Args:
            y_true: True edge labels.
            y_pred: Predicted edge labels.
            y_prob: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metrics.
        """
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {
            "accuracy": accuracy_score(y_true_np, y_pred_np),
            "f1": f1_score(y_true_np, y_pred_np),
            "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
            "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        }
        
        # Add AUC and AP if probabilities are provided
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy()
            metrics["auc"] = roc_auc_score(y_true_np, y_prob_np)
            metrics["ap"] = average_precision_score(y_true_np, y_prob_np)
        
        return metrics
    
    @staticmethod
    def community_detection_metrics(
        true_communities: List[set],
        pred_communities: List[set],
    ) -> Dict[str, float]:
        """Calculate community detection metrics.
        
        Args:
            true_communities: True community assignments.
            pred_communities: Predicted community assignments.
            
        Returns:
            Dictionary of metrics.
        """
        # Convert to label arrays
        all_nodes = set()
        for community in true_communities:
            all_nodes.update(community)
        for community in pred_communities:
            all_nodes.update(community)
        
        all_nodes = list(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        true_labels = np.zeros(len(all_nodes))
        pred_labels = np.zeros(len(all_nodes))
        
        for i, community in enumerate(true_communities):
            for node in community:
                if node in node_to_idx:
                    true_labels[node_to_idx[node]] = i
        
        for i, community in enumerate(pred_communities):
            for node in community:
                if node in node_to_idx:
                    pred_labels[node_to_idx[node]] = i
        
        metrics = {
            "nmi": normalized_mutual_info_score(true_labels, pred_labels),
            "ari": adjusted_rand_score(true_labels, pred_labels),
        }
        
        return metrics
    
    @staticmethod
    def graph_statistics(data) -> Dict[str, Union[int, float]]:
        """Calculate basic graph statistics.
        
        Args:
            data: PyTorch Geometric data object.
            
        Returns:
            Dictionary of graph statistics.
        """
        # Convert to NetworkX for easier analysis
        G = to_networkx(data, to_undirected=True)
        
        stats = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
            "assortativity": nx.degree_assortativity_coefficient(G),
        }
        
        # Add path-based metrics if graph is connected
        if nx.is_connected(G):
            stats["average_shortest_path_length"] = nx.average_shortest_path_length(G)
            stats["diameter"] = nx.diameter(G)
        else:
            stats["average_shortest_path_length"] = float("inf")
            stats["diameter"] = float("inf")
        
        return stats


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            device: Device to use for evaluation.
        """
        self.device = device or torch.device("cpu")
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_node_classification(
        self,
        model: torch.nn.Module,
        data,
        test_mask: torch.Tensor,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, Tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate node classification model.
        
        Args:
            model: Trained model.
            data: PyTorch Geometric data object.
            test_mask: Test node mask.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of metrics and optionally predictions.
        """
        model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            test_mask = test_mask.to(self.device)
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Get test predictions
            test_out = out[test_mask]
            test_y = data.y[test_mask]
            
            # Get predictions and probabilities
            y_pred = test_out.argmax(dim=1)
            y_prob = torch.softmax(test_out, dim=1)
            
            # Calculate metrics
            metrics = self.metrics_calculator.node_classification_metrics(
                test_y, y_pred, y_prob
            )
            
            if return_predictions:
                metrics["predictions"] = (y_pred, y_prob)
        
        return metrics
    
    def evaluate_link_prediction(
        self,
        model: torch.nn.Module,
        data,
        test_edge_index: torch.Tensor,
        test_edge_label: torch.Tensor,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, Tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate link prediction model.
        
        Args:
            model: Trained model.
            data: PyTorch Geometric data object.
            test_edge_index: Test edge indices.
            test_edge_label: Test edge labels.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of metrics and optionally predictions.
        """
        model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            test_edge_index = test_edge_index.to(self.device)
            test_edge_label = test_edge_label.to(self.device)
            
            # Forward pass
            out = model(data.x, data.edge_index, test_edge_index)
            
            # Get predictions and probabilities
            y_pred = (torch.sigmoid(out) > 0.5).long()
            y_prob = torch.sigmoid(out)
            
            # Calculate metrics
            metrics = self.metrics_calculator.link_prediction_metrics(
                test_edge_label, y_pred, y_prob
            )
            
            if return_predictions:
                metrics["predictions"] = (y_pred, y_prob)
        
        return metrics
    
    def evaluate_community_detection(
        self,
        model: torch.nn.Module,
        data,
        true_communities: List[set],
        return_communities: bool = False,
    ) -> Dict[str, Union[float, List[set]]]:
        """Evaluate community detection model.
        
        Args:
            model: Trained model.
            data: PyTorch Geometric data object.
            true_communities: True community assignments.
            return_communities: Whether to return predicted communities.
            
        Returns:
            Dictionary of metrics and optionally communities.
        """
        model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Get node embeddings
            embeddings = out.cpu().numpy()
            
            # Cluster embeddings (simple k-means)
            from sklearn.cluster import KMeans
            
            n_communities = len(true_communities)
            kmeans = KMeans(n_clusters=n_communities, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Convert to community format
            pred_communities = []
            for i in range(n_communities):
                community = set()
                for node_idx, label in enumerate(cluster_labels):
                    if label == i:
                        community.add(node_idx)
                pred_communities.append(community)
            
            # Calculate metrics
            metrics = self.metrics_calculator.community_detection_metrics(
                true_communities, pred_communities
            )
            
            if return_communities:
                metrics["communities"] = pred_communities
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        data,
        test_mask: torch.Tensor,
        task: str = "node_classification",
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names and models.
            data: PyTorch Geometric data object.
            test_mask: Test mask.
            task: Task type ('node_classification', 'link_prediction').
            
        Returns:
            Dictionary of results for each model.
        """
        results = {}
        
        for name, model in models.items():
            if task == "node_classification":
                results[name] = self.evaluate_node_classification(model, data, test_mask)
            elif task == "link_prediction":
                # This would need test_edge_index and test_edge_label
                raise NotImplementedError("Link prediction comparison not implemented")
            else:
                raise ValueError(f"Unknown task: {task}")
        
        return results
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "accuracy",
    ) -> List[Tuple[str, float]]:
        """Create a leaderboard from results.
        
        Args:
            results: Results dictionary.
            metric: Metric to rank by.
            
        Returns:
            List of (model_name, score) tuples sorted by score.
        """
        leaderboard = []
        
        for model_name, metrics in results.items():
            if metric in metrics:
                leaderboard.append((model_name, metrics[metric]))
        
        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        return leaderboard
