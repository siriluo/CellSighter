import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, AddSelfLoops
from torch_geometric.utils import dropout_adj
import numpy as np
from typing import Optional, Union, Tuple


class GATv2Model(nn.Module):
    """
    GATv2 (Graph Attention Network v2) implementation using PyTorch Geometric.
    
    This model supports both node-level and graph-level prediction tasks.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        task_type: str = "node",  # "node" or "graph"
        residual: bool = True,
        normalize: bool = True
    ):
        super(GATv2Model, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.task_type = task_type
        self.residual = residual
        self.normalize = normalize

        # Input projection layer
        # Initial linear layer
        self.input_lin_layer = nn.Linear(num_features, hidden_dim)
        
        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList() if residual else None
        self.norm_layers = nn.ModuleList() if normalize else None
        
        for i in range(num_layers):
            # For the last layer, we might want to reduce to single head
            heads = num_heads if i < num_layers - 1 else 1
            out_channels = hidden_dim if i < num_layers - 1 else hidden_dim
            
            gat_layer = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=out_channels,
                heads=heads,
                dropout=attention_dropout,
                concat=True if i < num_layers - 1 else False,
                share_weights=False,
                bias=True
            )
            
            self.gat_layers.append(gat_layer)
            
            # Residual connections
            if residual and i > 0:
                if i < num_layers - 1:
                    residual_dim = hidden_dim * heads
                else:
                    residual_dim = hidden_dim
                self.residual_layers.append(
                    nn.Linear(hidden_dim, residual_dim) if hidden_dim != residual_dim else nn.Identity()
                )
            
            # Layer normalization
            if normalize:
                if i < num_layers - 1:
                    norm_dim = hidden_dim * heads
                else:
                    norm_dim = hidden_dim
                self.norm_layers.append(nn.LayerNorm(norm_dim))
        
        # Graph-level pooling (if task_type is "graph")
        if task_type == "graph":
            self.pool = global_mean_pool  # Can also use global_max_pool or attention pooling
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass of the GATv2 model.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level tasks [num_nodes]
            edge_attr: Edge attributes (not used in basic GATv2 but can be extended)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Node/graph predictions and optionally attention weights
        """
        attention_weights = [] if return_attention_weights else None
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store for residual connections
        residual = x
        
        # Apply GATv2 layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply attention layer
            if return_attention_weights:
                x_new, (edge_index_att, att_weights) = gat_layer(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_index_att, att_weights))
            else:
                x_new = gat_layer(x, edge_index)
            
            # Residual connection (skip first layer)
            if self.residual and i > 0:
                if self.residual_layers[i-1] is not None:
                    residual = self.residual_layers[i-1](residual)
                x_new = x_new + residual
            
            # Layer normalization
            if self.normalize:
                x_new = self.norm_layers[i](x_new)
            
            # Activation and dropout (except for last layer)
            if i < len(self.gat_layers) - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            x = x_new
            residual = x
        
        # Graph-level pooling for graph classification
        if self.task_type == "graph":
            if batch is None:
                # Single graph case
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.pool(x, batch)
        
        # Final prediction layers
        x = self.output_layers(x)
        
        if return_attention_weights:
            return x, attention_weights
        else:
            return x
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        layer_idx: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights from a specific layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Edge indices and attention weights
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(
                x, edge_index, return_attention_weights=True
            )
            return attention_weights[layer_idx]

