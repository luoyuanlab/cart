## Linformer paper: https://arxiv.org/abs/2006.04768
## Blog: https://sh-tsang.medium.com/brief-review-linformer-self-attention-with-linear-complexity-d87fce25fe8f
## This reduces the attention complexity from O(L^2) to O(L)

from linformer import Linformer
import torch.nn as nn
    
class LinformerTransformer(nn.Module):
    def __init__(self, num_layers, dim, seq_len, heads, k):
        super().__init__()
        self.layer = Linformer(
                dim=dim,           ## Input dimension
                seq_len=seq_len,   ## Sequence length
                depth=num_layers,           ## Depth per block
                heads=heads,       ## Number of attention heads
                k=k                ## Low-rank approximation
        )
    def forward(self, x):
        x = self.layer(x) + x  ## Residual connection
        #x = self.layer(x)
        return x