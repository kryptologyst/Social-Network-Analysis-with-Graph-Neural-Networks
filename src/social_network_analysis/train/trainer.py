"""Training utilities for social network analysis models."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from ..utils.device import get_device


class Trainer:
    """Trainer class for GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        patience: int = 10,
        save_dir: str = "checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to use for training.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            optimizer: Optimizer type ('adam', 'adamw').
            scheduler: Learning rate scheduler ('reduce_on_plateau', 'cosine').
            patience: Patience for early stopping.
            save_dir: Directory to save checkpoints.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        # Optimizer
        if optimizer == "adam":
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Scheduler
        self.scheduler = None
        if scheduler == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=patience // 2)
        elif scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def train_epoch(
        self,
        data,
        train_mask: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            data: PyTorch Geometric data object.
            train_mask: Training node mask.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        
        # Compute loss
        loss = criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        pred = out[train_mask].argmax(dim=1)
        acc = (pred == data.y[train_mask]).float().mean().item()
        
        return loss.item(), acc
    
    def validate(
        self,
        data,
        val_mask: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            data: PyTorch Geometric data object.
            val_mask: Validation node mask.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            val_mask = val_mask.to(self.device)
            
            # Forward pass
            out = self.model(data.x, data.edge_index)
            
            # Compute loss
            loss = criterion(out[val_mask], data.y[val_mask])
            
            # Compute accuracy
            pred = out[val_mask].argmax(dim=1)
            acc = (pred == data.y[val_mask]).float().mean().item()
        
        return loss.item(), acc
    
    def train(
        self,
        data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        criterion: nn.Module,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            data: PyTorch Geometric data object.
            train_mask: Training node mask.
            val_mask: Validation node mask.
            criterion: Loss function.
            epochs: Number of training epochs.
            verbose: Whether to print progress.
            
        Returns:
            Training history dictionary.
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            # Train
            train_loss, train_acc = self.train_epoch(data, train_mask, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(data, val_mask, criterion)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_acc"].append(val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        
        return self.training_history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.training_history = checkpoint["training_history"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def evaluate(
        self,
        data,
        test_mask: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float, torch.Tensor]:
        """Evaluate the model on test set.
        
        Args:
            data: PyTorch Geometric data object.
            test_mask: Test node mask.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, accuracy, predictions).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            test_mask = test_mask.to(self.device)
            
            # Forward pass
            out = self.model(data.x, data.edge_index)
            
            # Compute loss
            loss = criterion(out[test_mask], data.y[test_mask])
            
            # Compute accuracy
            pred = out[test_mask].argmax(dim=1)
            acc = (pred == data.y[test_mask]).float().mean().item()
        
        return loss.item(), acc, pred


class LinkPredictionTrainer:
    """Trainer for link prediction tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        optimizer: str = "adam",
        save_dir: str = "checkpoints",
    ):
        """Initialize link prediction trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to use for training.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            optimizer: Optimizer type ('adam', 'adamw').
            save_dir: Directory to save checkpoints.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        # Optimizer
        if optimizer == "adam":
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_loss = float("inf")
        self.training_history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    
    def train_epoch(
        self,
        data,
        train_edge_index: torch.Tensor,
        train_edge_label: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            data: PyTorch Geometric data object.
            train_edge_index: Training edge indices.
            train_edge_label: Training edge labels.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, auc).
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        train_edge_index = train_edge_index.to(self.device)
        train_edge_label = train_edge_label.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index, train_edge_index)
        
        # Compute loss
        loss = criterion(out, train_edge_label.float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute AUC (simplified)
        pred_prob = torch.sigmoid(out)
        auc = self._compute_auc(pred_prob, train_edge_label)
        
        return loss.item(), auc
    
    def validate(
        self,
        data,
        val_edge_index: torch.Tensor,
        val_edge_label: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            data: PyTorch Geometric data object.
            val_edge_index: Validation edge indices.
            val_edge_label: Validation edge labels.
            criterion: Loss function.
            
        Returns:
            Tuple of (loss, auc).
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            val_edge_index = val_edge_index.to(self.device)
            val_edge_label = val_edge_label.to(self.device)
            
            # Forward pass
            out = self.model(data.x, data.edge_index, val_edge_index)
            
            # Compute loss
            loss = criterion(out, val_edge_label.float())
            
            # Compute AUC
            pred_prob = torch.sigmoid(out)
            auc = self._compute_auc(pred_prob, val_edge_label)
        
        return loss.item(), auc
    
    def _compute_auc(self, pred_prob: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute AUC score.
        
        Args:
            pred_prob: Predicted probabilities.
            labels: True labels.
            
        Returns:
            AUC score.
        """
        from sklearn.metrics import roc_auc_score
        
        return roc_auc_score(labels.cpu().numpy(), pred_prob.cpu().numpy())
    
    def train(
        self,
        data,
        train_edge_index: torch.Tensor,
        train_edge_label: torch.Tensor,
        val_edge_index: torch.Tensor,
        val_edge_label: torch.Tensor,
        criterion: nn.Module,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            data: PyTorch Geometric data object.
            train_edge_index: Training edge indices.
            train_edge_label: Training edge labels.
            val_edge_index: Validation edge indices.
            val_edge_label: Validation edge labels.
            criterion: Loss function.
            epochs: Number of training epochs.
            verbose: Whether to print progress.
            
        Returns:
            Training history dictionary.
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            # Train
            train_loss, train_auc = self.train_epoch(data, train_edge_index, train_edge_label, criterion)
            
            # Validate
            val_loss, val_auc = self.validate(data, val_edge_index, val_edge_label, criterion)
            
            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_auc"].append(train_auc)
            self.training_history["val_auc"].append(val_auc)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
                )
        
        return self.training_history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]
