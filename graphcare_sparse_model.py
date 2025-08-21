import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, GINConv
from torch_geometric.nn.inits import reset

from typing import Callable, Optional, Union

import torch
import random
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax
from torch.nn import LeakyReLU


class BiAttentionGNNConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 edge_attn=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn
        if edge_attn:
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.nn, 'reset_parameters'):
            self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, attn: Tensor = None,
                edge_weights: Tensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Always call propagate with consistent arguments
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, 
                           attn=attn, edge_weights=edge_weights)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None and edge_attr is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: OptTensor = None, attn: OptTensor = None, 
                edge_weights: OptTensor = None) -> Tensor:
        # Base message
        if self.edge_attn and edge_attr is not None and self.W_R is not None:
            w_rel = self.W_R(edge_attr)
            if attn is not None:
                msg = (x_j * attn + w_rel * edge_attr).relu()
            else:
                msg = (x_j + w_rel * edge_attr).relu()
        else:
            if attn is not None:
                msg = (x_j * attn).relu()
            else:
                msg = x_j.relu()
        
        # Apply edge weights for sparsification
        if edge_weights is not None:
            msg = msg * edge_weights.view(-1, 1)
        
        return msg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class EdgeScorer(nn.Module):
    """Module to learn edge importance scores for soft sparsification"""
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super(EdgeScorer, self).__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # node_i + node_j + edge features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            edge_scores: Edge importance scores [num_edges, 1]
        """
        # Get source and target node features
        src_nodes = x[edge_index[0]]  # [num_edges, node_dim]
        tgt_nodes = x[edge_index[1]]  # [num_edges, node_dim]
        
        # Transform features
        src_transformed = self.node_transform(src_nodes)  # [num_edges, hidden_dim]
        tgt_transformed = self.node_transform(tgt_nodes)  # [num_edges, hidden_dim]
        edge_transformed = self.edge_transform(edge_attr)  # [num_edges, hidden_dim]
        
        # Concatenate features
        combined_features = torch.cat([
            src_transformed, 
            tgt_transformed, 
            edge_transformed
        ], dim=1)  # [num_edges, hidden_dim * 3]
        
        # Score edges
        edge_scores = self.scorer(combined_features)  # [num_edges, 1]
        
        return edge_scores


class SparseGraphCare(nn.Module):
    def __init__(
            self, num_nodes, num_rels, max_visit, embedding_dim, hidden_dim, 
            out_channels, layers=3, dropout=0.5, decay_rate=0.03, node_emb=None, rel_emb=None,
            freeze=False, patient_mode="joint", use_alpha=True, use_beta=True, use_edge_attn=True, 
            self_attn=0., gnn="BAT", attn_init=None, drop_rate=0.,
            # New sparsification parameters
            use_sparsification=True, sparsification_ratio=0.1, l1_lambda=1e-4, connectivity_lambda=1e-3
        ):
        super(SparseGraphCare, self).__init__()

        self.gnn = gnn
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.edge_attn = use_edge_attn
        self.drop_rate = drop_rate
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit

        # Sparsification parameters
        self.use_sparsification = use_sparsification
        self.sparsification_ratio = sparsification_ratio
        self.l1_lambda = l1_lambda
        self.connectivity_lambda = connectivity_lambda

        # Decay weights for visit importance
        j = torch.arange(max_visit).float()
        self.lambda_j = torch.exp(self.decay_rate * (max_visit - j)).unsqueeze(0).reshape(1, max_visit, 1).float()

        # Embeddings
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)

        # Linear transformation and normalization
        self.lin = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layers = layers
        self.dropout = dropout

        # Initialize attention and convolution layers
        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize EdgeScorer for sparsification
        if self.use_sparsification:
            self.edge_scorer = EdgeScorer(hidden_dim, hidden_dim, hidden_dim // 2)

        # Build layers
        for layer in range(1, layers+1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(num_nodes, num_nodes)
                if attn_init is not None:
                    attn_init = attn_init.float()
                    attn_init_matrix = torch.eye(num_nodes).float() * attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(attn_init_matrix)
                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
                    
            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)
                
            if self.gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(
                    nn.Linear(hidden_dim, hidden_dim), 
                    edge_dim=hidden_dim, 
                    edge_attn=self.edge_attn, 
                    eps=self_attn
                )
            elif self.gnn == "GAT":
                self.conv[str(layer)] = GATConv(hidden_dim, hidden_dim)
            elif self.gnn == "GIN":
                self.conv[str(layer)] = GINConv(nn.Linear(hidden_dim, hidden_dim))

        # Final prediction layer
        if self.patient_mode == "joint":
            self.MLP = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.MLP = nn.Linear(hidden_dim, out_channels)

    def to(self, device):
        super().to(device)
        self.lambda_j = self.lambda_j.float().to(device)
        return self

    def compute_sparsification_loss(self, edge_scores, edge_index):
        """Compute sparsification regularization losses"""
        
        # L1 sparsity loss (encourage sparsity)
        l1_loss = torch.mean(edge_scores)
        
        # Connectivity preservation loss (ensure graph remains connected)
        # Count edges per node
        num_nodes = torch.max(edge_index) + 1
        edge_counts = torch.zeros(num_nodes, device=edge_scores.device)
        edge_counts.scatter_add_(0, edge_index[0], edge_scores.squeeze())
        
        # Penalize nodes with too few weighted edges
        min_edge_threshold = 1.0  # Each node should have at least this much total edge weight
        connectivity_loss = torch.mean(torch.relu(min_edge_threshold - edge_counts))
        
        total_loss = self.l1_lambda * l1_loss + self.connectivity_lambda * connectivity_loss
        
        return total_loss, l1_loss, connectivity_loss

    def forward(self, node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes, store_attn=False, in_drop=False):
        
        # Original edge dropping for data augmentation
        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            if edges_to_remove > 0:
                indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
                edge_mask = torch.tensor([i not in indices_to_remove for i in range(edge_count)], 
                                       device=edge_index.device)
                edge_index = edge_index[:, edge_mask]
                rel_ids = rel_ids[edge_mask]

        # Embed nodes and relations
        x = self.node_emb(node_ids).float()
        edge_attr = self.rel_emb(rel_ids).float()

        # Transform to hidden dimension
        x = self.lin(x)
        edge_attr = self.lin(edge_attr)

        # Compute edge scores for sparsification
        edge_scores = None
        sparsification_loss = 0.0
        if self.use_sparsification and self.training:
            edge_scores = self.edge_scorer(x, edge_index, edge_attr)
            sparsification_loss, l1_loss, connectivity_loss = self.compute_sparsification_loss(edge_scores, edge_index)

        # Store attention weights if requested
        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []
            self.edge_scores_history = []

        # Graph convolution layers
        for layer in range(1, self.layers+1):
            # Compute attention weights
            if self.use_alpha:
                alpha = torch.softmax(self.alpha_attn[str(layer)](visit_node.float()), dim=1)

            if self.use_beta:
                beta = torch.tanh(self.beta_attn[str(layer)](visit_node.float())) * self.lambda_j

            # Combine attention mechanisms
            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1), device=edge_index.device)
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes), device=edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes), device=edge_index.device)
                
            attn = torch.sum(attn, dim=1)
            
            # Map attention to edges
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn_edges = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            # Apply convolution with edge weights
            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn_edges, edge_weights=edge_scores)
            elif self.gnn == "GAT":
                x = self.conv[str(layer)](x, edge_index)
            elif self.gnn == "GIN":
                x = self.conv[str(layer)](x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Store attention weights if requested
            if store_attn:
                if self.use_alpha:
                    self.alpha_weights.append(alpha)
                if self.use_beta:
                    self.beta_weights.append(beta)
                self.attention_weights.append(attn_edges)
                if self.gnn == "BAT":
                    self.edge_weights.append(w_rel)
                if edge_scores is not None:
                    self.edge_scores_history.append(edge_scores.detach())

        # Patient representation aggregation
        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # Graph-level representation via global pooling
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # Node-level representation via EHR node averaging
            batch_size = batch.max().item() + 1
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i])
                for i in range(batch_size)
            ])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

        # Final prediction
        if self.patient_mode == "joint":
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
            logits = self.MLP(x_concat)
        elif self.patient_mode == "graph":
            logits = self.MLP(x_graph)
        elif self.patient_mode == "node":
            logits = self.MLP(x_node)

        # Return results
        if store_attn:
            return_dict = {
                'logits': logits,
                'alpha_weights': getattr(self, 'alpha_weights', []),
                'beta_weights': getattr(self, 'beta_weights', []),
                'attention_weights': getattr(self, 'attention_weights', [])
            }
            if self.gnn == "BAT":
                return_dict['edge_weights'] = getattr(self, 'edge_weights', [])
            if hasattr(self, 'edge_scores_history') and self.edge_scores_history:
                return_dict['edge_scores'] = self.edge_scores_history
            return return_dict
        else:
            if self.use_sparsification and self.training:
                return logits, sparsification_loss
            else:
                return logits


# Backward compatibility
class GraphCare(SparseGraphCare):
    def __init__(self, *args, **kwargs):
        # Disable sparsification for original GraphCare
        kwargs['use_sparsification'] = False
        super().__init__(*args, **kwargs)