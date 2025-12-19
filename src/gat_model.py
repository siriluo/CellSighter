import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, AddSelfLoops
from torch_geometric.utils import dropout_adj, softmax
import numpy as np
from typing import Optional, Union, Tuple
# from einops import rearrange



model_dict = {
    'resnet18': [512],
    'resnet34': [512],
    'resnet50': [2048],
    'resnet101': [2048],
}
    

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e) # B N N
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayerMod(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerMod, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        # Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (N, out_features)
        # e = self._prepare_attentional_mechanism_input(Wh)

        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [num_nodes, out_features]
        
        # Get source and target node indices
        edge_index_i = edge_index[0]  # Source nodes
        edge_index_j = edge_index[1]  # Target nodes
        
        # Compute attention coefficients
        Wh_i = torch.matmul(Wh, self.a[:self.out_features, :])  # [num_nodes, 1]
        Wh_j = torch.matmul(Wh, self.a[self.out_features:, :])  # [num_nodes, 1]
        
        # Attention for each edge
        e = Wh_i[edge_index_i] + Wh_j[edge_index_j]  # [num_edges, 1]
        e = self.leakyrelu(e)
        

        # zero_vec = -9e15*torch.ones_like(e) # B N N
        # attention = torch.where(adj > 0, e, zero_vec)

        # Normalize attention coefficients using softmax
        # PyG's softmax handles per-node normalization automatically
        attention = softmax(e, edge_index_j, num_nodes=h.size(0))  # [num_edges, 1]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


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


class GATv2ClassificationHead(nn.Module):
    def __init__(self,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        attention_dropout: float = 0.2,
        task_type: str = "node",  # "node" or "graph"
        residual: bool = True,
        normalize: bool = True,
        name='resnet50',
        **kwargs):
        super(GATv2ClassificationHead, self).__init__(**kwargs)

        feat_dim = model_dict[name][0]

        heads = num_heads # if i < num_layers - 1 else 1

        self.feature_norm = nn.LayerNorm(feat_dim)

        self.gat_layer = GATv2Conv(
            in_channels=feat_dim,
            out_channels=feat_dim,
            heads=heads,
            dropout=dropout_rate, # attention_dropout
            concat=False, # if i < num_layers - 1 else False
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(dropout_rate),
            # nn.Linear(input_dim, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, num_classes)
        )



    def forward(self,         
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False) -> torch.Tensor:

        attention_weights = []

        features = self.feature_norm(x)

        graph_feats = self.gat_layer(x, edge_index)
        if return_attention_weights:
            graph_feats, (edge_index_att, att_weights) = self.gat_layer(
                x, edge_index, return_attention_weights=True
            )
            attention_weights.append((edge_index_att, att_weights))
        else:
            graph_feats = self.gat_layer(x, edge_index)

        probs = self.classifier(graph_feats)

        return probs


