## from scGPT (https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py)
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import scipy.sparse as sp
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import numpy as np



class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,    ## number of batches
        embedding_dim: int,     ## batch dimension
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor: ## x: (batch_size)
        x = self.embedding(x)  ## (batch, embsize)
        x = self.enc_norm(x)
        return x


class ExpValueEncoder(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0, max_value: float = np.inf):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.max_value = max_value
        
        self.expression_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        ## TODO: test using actual embedding layer if input is categorical
        ## expand last dimension
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value) ## clip x to [-inf, max_value]
        expr_emb = self.expression_embedding(x)
        
        return self.dropout(x)

    
## weighted_MSELoss
class weighted_MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, weights):
        if self.reduction == 'mean':
            return (((inputs - targets)**2 ) * weights).sum() / weights.sum()
        else:
            return (((inputs - targets)**2 ) * weights).sum()
        
        
## adjacency matrix normalization, from VGAE paper(https://github.com/DaehanKim/vgae_pytorch/tree/master)
## extract coor, values and shape of a sparse matrix
def sparse_to_tuple(sparse_mx, clip_min=None, clip_max=None):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    if clip_min != None or clip_max != None:
        values = np.clip(values, a_min=clip_min, a_max=clip_max)
    shape = sparse_mx.shape
    return coords, values, shape


def get_weight_matrix(features, power=1):
    
    n_nodes, feat_RNA_dim = features.shape
    
    features = sp.csr_matrix(features)
    features = sparse_to_tuple(features)
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))

    #loss_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - torch.sparse.sum(features)) * 2)
    pos_weight = float(features.shape[0] * features.shape[1] - len(features.coalesce().values())) / len(features.coalesce().values())
    pos_weight = pos_weight * power
    print(pos_weight)
    if pos_weight > 1:
        weight_mask = features.to_dense().view(-1) != 0
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight ## non-zero positions marked as pos_weight, zero positions marked as 1
    else:
        weight_tensor = torch.ones_like(features)
    weight_tensor = weight_tensor.view(n_nodes, feat_RNA_dim)
    
    return weight_tensor