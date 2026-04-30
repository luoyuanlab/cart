import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class AttentionLayer(nn.Module):
    
    def __init__(self, in_feat):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(in_feat, 1) 
        self.fc = nn.Linear(in_feat, 1) 

    def forward(self, x, mask=None):

        attention_scores = self.attention_weights(x) 
        if mask != None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf')) 
        attention_weights = F.softmax(attention_scores, dim=1) 
        x = torch.sum(attention_weights * x, dim=1, keepdim=True) 
        
        return torch.squeeze(x), attention_weights.squeeze()



class Gene_to_Cell_AttentionLayer(nn.Module):
    
    def __init__(self, in_feat, dropout=0.0, act=F.relu, mask_avg=False):
        super(Gene_to_Cell_AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.act = act
        self.mask_avg = mask_avg
        
        self.w_weights = nn.Linear(in_feat, in_feat) 
        self.u_weights = nn.Linear(in_feat, 1)        
        
    def forward(self, x, mask=None):
        
        v = self.w_weights(x) 
        v = F.tanh(v)
        attention_scores = self.u_weights(v) 
        if mask != None:
            attention_scores = attention_scores.squeeze().masked_fill(mask == 0, float('-inf')) 
            if self.mask_avg:
                attention_scores = attention_scores/(mask.sum(axis=0))
        attention_weights = F.softmax(attention_scores, dim=1) 
        x = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1, keepdim=True) 
    
        return torch.squeeze(x), attention_weights 

    

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


    
class HGAT_sparse(nn.Module): 

    def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
        super(HGAT_sparse, self).__init__()
        self.e_dropout = nn.Dropout(dropout)
        self.in_ch_n = in_ch_n
        self.out_ch = out_ch
        self.alpha = alpha
        self.concat = concat
        
        self.transfer = transfer

        if self.transfer:
            self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))
        else:
            self.register_parameter('wt', None)      
        
        if bias:
            self.bias = Parameter(torch.Tensor(1, self.out_ch))
        else:
            self.register_parameter('bias', None)       
        
        self.coarsen = coarsen

        self.reset_parameters()

    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.out_ch)
        if self.wt is not None:
            self.wt.data.uniform_(-stdv, stdv)     
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv) 

    def reset_parameters_xavier(self): 
        if self.wt is not None:
            nn.init.xavier_uniform_(self.wt)     
        
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)       

    def std_scale(self, x):
        xstd = x.std(1, unbiased=False, keepdim=True)
        xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device))
        x = (x - x.mean(1, keepdim=True)) / xstd
        return x

    def forward(self, x, xe, pair, a, val=None, e_degs=None, n_degs=None):
        
        if self.transfer:
            x = x.mm(self.wt) 
            xe = xe.mm(self.wt) 
            
            if self.bias is not None: 
                
                x = x + self.bias
                xe = xe + self.bias      

        n_edge = xe.shape[0] 
        n_node = x.shape[0] 
        
        if val is None:
            pair_h = xe[ pair[0] ] * x[ pair[1] ] 
        else:
            pair_h = xe[ pair[0] ] * x[ pair[1] ] * val 
            
        if e_degs is not None:
            pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
        if n_degs is not None:
            pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 
        pair_e = torch.mm(pair_h, a).squeeze() 
        
        
        e = torch.zeros(n_edge, n_node, device=pair.device) 
        e[pair[0], pair[1]] = torch.exp(pair_e) 
        e = torch.log(1e-10 + self.e_dropout(e)) 
        
        attention_edge = F.softmax(e, dim=1) 

        xe_out = torch.mm(attention_edge, x) 
        
        attention_node = F.softmax(e.transpose(0,1), dim=1) 

        x = torch.mm(attention_node, xe) 

        if self.concat:
            x = F.elu(x)
            xe_out = F.elu(xe_out)
        else:
            x = F.relu(x)
            xe_out = F.relu(xe_out)
        
        if self.coarsen:
            return x, xe_out, torch.exp(e.T) 
        else:
            return x, xe_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'
