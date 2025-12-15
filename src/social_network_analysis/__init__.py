"""Package initialization for social network analysis."""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .data.dataset import SocialNetworkDataset
from .models.gnn_models import GCN, GraphSAGE, GAT, GraphClassifier, LinkPredictor
from .train.trainer import Trainer, LinkPredictionTrainer
from .eval.evaluator import ModelEvaluator, MetricsCalculator
from .utils.device import get_device, set_seed, count_parameters, get_model_size_mb

__all__ = [
    "SocialNetworkDataset",
    "GCN",
    "GraphSAGE", 
    "GAT",
    "GraphClassifier",
    "LinkPredictor",
    "Trainer",
    "LinkPredictionTrainer",
    "ModelEvaluator",
    "MetricsCalculator",
    "get_device",
    "set_seed",
    "count_parameters",
    "get_model_size_mb",
]
