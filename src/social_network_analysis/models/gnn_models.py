"""Graph Neural Network models for social network analysis."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool


class GCN(nn.Module):
    """Graph Convolutional Network for node classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Node predictions.
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = "mean",
    ):
        """Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of SAGE layers.
            dropout: Dropout rate.
            aggregator: Aggregation method ('mean', 'max', 'lstm').
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # SAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Node predictions.
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GAT(nn.Module):
    """Graph Attention Network for node classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_batch_norm = use_batch_norm
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    output_dim,
                    heads=1,
                    dropout=dropout,
                )
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Node predictions.
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphClassifier(nn.Module):
    """Graph-level classifier using node embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        pooling: str = "mean",
        dropout: float = 0.5,
    ):
        """Initialize graph classifier.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GCN layers.
            pooling: Pooling method ('mean', 'max', 'sum').
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.pooling = pooling
        
        # GCN layers for node embeddings
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for graph-level tasks.
            
        Returns:
            torch.Tensor: Graph predictions.
        """
        # Get node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.convs[-1](x, edge_index)
        
        # Graph-level pooling
        if batch is not None:
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)  # Default to mean
        else:
            # Single graph
            if self.pooling == "mean":
                x = x.mean(dim=0, keepdim=True)
            elif self.pooling == "max":
                x = x.max(dim=0, keepdim=True)[0]
            else:
                x = x.mean(dim=0, keepdim=True)  # Default to mean
        
        # Classification
        x = self.classifier(x)
        
        return x


class LinkPredictor(nn.Module):
    """Link prediction model using node embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """Initialize link predictor.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # GCN layers for node embeddings
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Link prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices for message passing.
            edge_label_index: Edge indices for prediction.
            
        Returns:
            torch.Tensor: Link predictions.
        """
        # Get node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.convs[-1](x, edge_index)
        
        # Get embeddings for edge endpoints
        src_emb = x[edge_label_index[0]]
        dst_emb = x[edge_label_index[1]]
        
        # Concatenate embeddings
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        # Predict link probability
        pred = self.predictor(edge_emb)
        
        return pred.squeeze()
