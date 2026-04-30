import torch
import torch.nn.functional as F
from torch import Tensor, nn
import scipy.sparse as sp
import scanpy as sc
from typing import Optional, Union
import numpy as np



def preprocess(adata, ribosomal: Union['regress_out', 'remove', 'none'] = 'none'):
    
    ribo_genes = adata.var_names.str.startswith(("RPS","RPL"))
    adata.obs['percent_ribo'] = np.sum(adata[:, ribo_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1

    if ribosomal == 'regress_out': 
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_counts', 'percent_ribo'])
    elif ribosomal == 'remove':
        adata = adata[:, ~ribo_genes]
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
    elif ribosomal == 'none':
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
    else:
        raise ValueError("ribosomal_gene should only be 'regress_out' or 'remove'")
    return adata



def nonzero_median(arr, axis=0):
    if axis == 0:
        return np.array([np.median(col[col != 0]) if np.any(col != 0) else np.nan for col in arr.T])
    elif axis == 1:
        return np.array([np.median(row[row != 0]) if np.any(row != 0) else np.nan for row in arr])
    else:
        raise ValueError("axis must be 0 (column-wise) or 1 (row-wise)")
        
        
        
def data_scale(adata, scale_feature: Union['gene', 'cell', 'none'] = 'none', scale_type: Union['max', 'median', 'ndist'] = 'median'):
    if scale_feature == 'gene':
        if scale_type == 'max':
            adata.var['gene_max'] = adata.X.A.max(axis=0)
            adata.X = adata.X / adata.var['gene_max'].values
        elif scale_type == 'median':
            adata.var['gene_median'] = nonzero_median(adata.X.A, axis=0)
            adata.X = adata.X / adata.var['gene_median'].values
        elif scale_type == "ndist":
            sc.pp.scale(adata, max_value=10, zero_center=False, copy=False)
        else:
            raise ValueError("scale_type must be max, median or ndist")
    elif scale_feature == 'cell':
        adata.obs['max_cell_count'] = adata.X.max(axis=1).A.squeeze()
        adata.X = (adata.X.A.T / adata.obs['max_cell_count'].values).T
    else:
        pass
    return adata



class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor: 
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x



class ExpValueEncoder(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0, max_value: float = np.inf):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.max_value = max_value
        
        self.expression_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.expression_embedding(x)
        
        return x

    

class weighted_MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, weights):
        if self.reduction == 'mean':
            return (((inputs - targets)**2 ) * weights).sum() / weights.sum()
        else:
            return (((inputs - targets)**2 ) * weights).sum()
        
        

def sparse_to_tuple(sparse_mx, clip_min=None, clip_max=None):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    if clip_min != None or clip_max != None:
        values = np.clip(values, a_min=clip_min, a_max=clip_max)
    shape = sparse_mx.shape
    return coords, values, shape



def get_weight_matrix_pos(features, power=1):
    
    n_nodes, feat_RNA_dim = features.shape
    
    import scipy.sparse as sp
    import torch

    if not sp.issparse(features):
        features = sp.csr_matrix(features)

    features = sparse_to_tuple(features)
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
    features = features.coalesce()

    pos_weight = float(features.shape[0] * features.shape[1] - len(features.values())) / len(features.values())
    pos_weight = pos_weight * power
    if pos_weight > 1:
        return pos_weight
    else:
        return 1.0
