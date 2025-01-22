## Response model

## attention layer
#1: x * softmax(x * w)
#2: x.T * softmax(tanh(x * w) * u)
#3: x.T * softmax(tanh(x * w) * u)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, multilabel_confusion_matrix
from CART.layers import HGNN_fc
import copy

class AttentionLayer(nn.Module):
    
    def __init__(self, in_feat):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(in_feat, 1)  ## [n_patient, n_cell, n_hid] -> [n_patient, n_cell, 1]
        self.fc = nn.Linear(in_feat, 1)  ## [n_patient, n_cell, n_hid] -> [n_patient, n_cell, 1]

    def forward(self, x, mask=None):
        ## x shape: [n_patient, n_cell, n_hid], mask shape: [n_patient, n_cell, 1]

        ## Calculate attention scores
        attention_scores = self.attention_weights(x)  ## [n_patient, n_cell, n_hid] -> [n_patient, n_cell, 1]
        if mask != None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  ## Mask padded cells [n_patient, n_cell, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  ## Softmax along cell dimension, [n_patient, n_cell, 1]
        ## Weighted sum using attention weights
        x = torch.sum(attention_weights * x, dim=1, keepdim=True)  ## [n_patient, n_cell, 1] * [n_patient, n_cell, n_hid] --> [n_patient, n_cell, n_hid] -sum-> [n_patient, 1, n_hid]
        
        return torch.squeeze(x)  ## [n_patient, n_hid]
"""
class AttentionLayer(nn.Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act = act
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, x, mask=None):
        ## x shape: [n_patient, n_cell, n_hid], mask shape: [n_patient, n_cell, 1]
        self.x = x
        
        self.v = F.tanh(torch.matmul(self.x, self.w_omega)) ## [n_patient, n_hid, n_cell] * [in_feat(n_cell), out_feat(1)] --> [n_patient, n_hid, out_feat(1)]
        self.attention_scores = torch.matmul(self.v, self.u_omega)  ## [n_patient, n_cell, out_feat(1)] * [out_feat(1), 1] --> [n_patient, n_cell, 1]
        if mask != None:
            self.attention_scores = self.attention_scores.masked_fill(mask == 0, float('-inf'))  ## Mask padded cells [n_patient, n_cell, 1]
        self.attention_weights = F.softmax(torch.squeeze(self.attention_scores) + 1e-6) ## [n_patient, n_cell]
        X = torch.matmul(torch.transpose(self.x, 1, 2), torch.unsqueeze(self.attention_weights, -1)) ## [n_patient, n_hid, n_cell] * [n_patient, n_cell, 1] = [n_patient, n_hid, 1]
    
        return torch.squeeze(X) ## [n_patient, n_hid]

class AttentionLayer(nn.Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act = act
        
        self.w_weights = nn.Linear(in_feat, out_feat)  ## [n_patient, n_cell, n_hid] -> [n_patient, n_cell, 1]
        self.u_weights = nn.Linear(out_feat, 1)        
        
    def forward(self, x, mask=None):
        ## x shape: [n_patient, n_cell, n_hid], mask shape: [n_patient, n_cell, 1]
        
        v = self.w_weights(x)  ## [n_patient, n_cell, in_feat(n_hid)] -> [n_patient, n_cell, out_feat(n_hid)]
        v = F.tanh(v)
        attention_scores = self.u_weights(v) ## [n_patient, n_cell, out_feat(n_hid)] -> [n_patient, n_cell, 1]
        if mask != None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  ## Mask padded cells [n_patient, n_cell, 1]
        #attention_weights = F.softmax(torch.squeeze(attention_scores) + 1e-6) ## [n_patient, n_cell, 1]
        #x = torch.matmul(torch.transpose(x, 1, 2), torch.unsqueeze(attention_weights, -1)) ## [n_patient, n_hid, n_cell] * [n_patient, n_cell, 1] = [n_patient, n_hid, 1]
        attention_weights = F.softmax(attention_scores + 1e-6, dim=1) ## [n_patient, n_cell, 1]
        x = torch.sum(attention_weights * x, dim=1, keepdim=True)  ## [n_patient, n_cell, 1] * [n_patient, n_cell, n_hid] --> [n_patient, n_cell, n_hid] -sum-> [n_patient, 1, n_hid]
    
        return torch.squeeze(x) ## [n_patient, n_hid]
"""

class ResPredModel(nn.Module):
    def __init__(self, sg_hid, fc_dropout=0.2):
        super(ResPredModel, self).__init__()
        self.cl_response_fc = nn.Sequential(OrderedDict([
                                          ('linear1', HGNN_fc(sg_hid, sg_hid)),
                                          ('activate1', nn.LeakyReLU()),
                                          ('layernorm1', nn.LayerNorm(sg_hid)),
                                          ('Drop1', nn.Dropout(fc_dropout)),
                                          ('linear2', HGNN_fc(sg_hid, sg_hid)),
                                          ('activate2', nn.LeakyReLU()),
                                          ('layernorm2', nn.LayerNorm(sg_hid)),
                                          ('Drop2', nn.Dropout(fc_dropout)),
                                          ('linear3', HGNN_fc(sg_hid, 2)),
                                         ]))
    def forward(self, x):
        x = self.cl_response_fc(x)
        return x


"""
## Function to rank cells by predicted response probability
def rank_cells_by_response(cell_output, cell_num=None):
    ## cell_output: [n_patient, n_cell, 2]
    probs = F.softmax(cell_output, dim=-1)[..., 1]  # Probability of Response, shape [n_patient, n_cell], rank is based on the Response probability
    sorted_indices = probs.argsort(dim=-1, descending=True)  # Sort cells per patient
    ranked_cells = torch.gather(cell_output, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, cell_output.size(-1))) # use torch.gather to reorder (or gather) values from cell_output based on dim=1
    
    ## Create mask for attention layer, marking real cells as 1, padding as 0
    if cell_num != None:
        mask = torch.arange(cell_output.size(1)).expand(cell_output.size(0), -1) < cell_num.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(cell_output.device)  # [n_patient, n_cell, 1]
    else:
        mask = None
    
    return ranked_cells, mask



def patient_batch_split(patient_cells, batch_size):
    cell_batches = []

    n_cells = len(patient_cells)
    n_batches = (n_cells + batch_size - 1) // batch_size  # Calculate how many batches per patient
    
    for j in range(n_batches):
        start_idx = j * batch_size
        end_idx = min((j + 1) * batch_size, n_cells)
        cell_batch = patient_cells[start_idx: end_idx]
        cell_batches.append(cell_batch)
        
    return cell_batches



class PatientResponseModel(nn.Module):
    def __init__(self, input_dim, max_cell):
        super(PatientResponseModel, self).__init__()
        self.cell_level_model = CellLevelModel(input_dim)  # Predict cell-level response
        self.attention_layer = RankedAttentionLayer()  # Aggregate cell responses at patient level
        self.max_cell = max_cell

    def rank_cells_by_response(self, cell_output, cell_num=None):
        # Compute response probability for each cell
        probs = F.softmax(cell_output, dim=-1)[..., 1]  # Probability of Response, shape [n_patient, n_cell], rank is based on the Response probability
        sorted_indices = probs.argsort(dim=-1, descending=True)  # Sort cells per patient by response probability
        # Gather cells by sorted indices to rank them
        ranked_cells = torch.gather(cell_output, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, cell_output.size(-1))) # use torch.gather to reorder (or gather) values from cell_output based on dim=1
        
        # Create a mask for real cells (1) and padded cells (0)
        if cell_num != None:
            mask = torch.arange(cell_output.size(1)).expand(cell_output.size(0), -1) < cell_num#.unsqueeze(1)
            mask = mask.unsqueeze(-1).to(cell_output.device)  # [n_patient, n_cell, 1]
        else:
            mask = None
        
        return ranked_cells, mask

    def patient_batch_split(self, patient_cells, batch_size):
        cell_batches = []

        n_cells = len(patient_cells)
        n_batches = (n_cells + batch_size - 1) // batch_size  # Calculate how many batches per patient

        for j in range(n_batches):
            start_idx = j * batch_size
            end_idx = min((j + 1) * batch_size, n_cells)
            cell_batch = patient_cells[start_idx: end_idx]
            cell_batches.append(cell_batch)

        return cell_batches

    def forward(self, m, pa_idx, 
                input_option: Union["embed", "embed_annotation"] = "embed", 
                cell_batch_size=64):
        
        # Step 1: Cell-level prediction
        cell_batches = self.patient_batch_split(pa_idx, cell_batch_size)
        cell_num = len(pa_idx)

        pa_cell_pred = torch.zeros(1, self.max_cell, 2)
        i = 0
        #for idx in tqdm(cell_batches, leave=False, desc='Cell level prediction:'):
        for idx in cell_batches:
            if input_option == "embed":
                sub_sgs = torch.Tensor(m[np.array(idx).tolist()].obsm['cell_embed']).unsqueeze(0)#.to(device)
            elif input_option == "embed_annotation":
                sub_sgs = torch.cat([torch.Tensor(m[np.array(idx).tolist()].obsm['cell_embed']),
                                     torch.Tensor(m[np.array(idx).tolist()].obsm['type_prob'])
                                    ], dim=1).unsqueeze(0)
            #cell_output = self.cell_level_model(sub_sgs.view(-1, sub_sgs.size(-1))).view(sub_sgs.size(0), sub_sgs.size(1), -1)
            cell_output = self.cell_level_model(sub_sgs)
            #pa_cell_pred = torch.cat([pa_cell_pred, cell_output], dim=1)
            pa_cell_pred[:1, i:i+len(idx), :] = cell_output
            i += len(idx)
        
        # Step 2: Rank cells by response probability
        ranked_cells, mask = self.rank_cells_by_response(pa_cell_pred, cell_num)
        
        # Step 3: Aggregate to patient-level response with attention
        patient_output = self.attention_layer(ranked_cells, mask).squeeze(1)  # Final shape: [n_patient, 2]
        
        return patient_output
"""

class PatientResponseModel(nn.Module):
    def __init__(self, input_dim, max_cell, fc_dropout=0, seed=None):
        super(PatientResponseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)                  ## Seed for PyTorch CPU
            torch.cuda.manual_seed(seed)             ## Seed for PyTorch GPU (single-GPU)
            torch.cuda.manual_seed_all(seed)        ## Seed for PyTorch GPU (multi-GPU)
            np.random.seed(seed)                    ## Seed for NumPy
            random.seed(seed)                       ## Seed for Python's random module
            
        self.attention_layer = AttentionLayer(input_dim)  ## Aggregate cell responses at patient level
        #self.attention_layer = AttentionLayer(input_dim, input_dim)  ## Aggregate cell responses at patient level
        self.res_pred_model = ResPredModel(input_dim, fc_dropout=fc_dropout)  # Predict cell-level response
        self.max_cell = max_cell


    def forward(self, m, pa_idx, device,
                input_option: Union["embed", "embed_annotation"] = "embed"):

        cell_num = len(pa_idx)
        
        if input_option == "embed":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1]).to(device)
            sgs = torch.Tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed']).unsqueeze(0).to(device)
            pa_cell_atten[:1, :cell_num, :] = sgs.to(device)
        elif input_option == "embed_annotation":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1] + m.obsm['type_prob'].shape[1]).to(device)
            sgs = torch.cat([torch.Tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed']),
                             torch.Tensor(m[np.array(pa_idx).tolist()].obsm['type_prob'])
                            ], dim=1).unsqueeze(0).to(device)
            pa_cell_atten[:1, :cell_num, :] = sgs.to(device)
            
        if cell_num != None:
            mask = torch.arange(pa_cell_atten.size(1)).expand(pa_cell_atten.size(0), -1) < cell_num#.unsqueeze(1)
            mask = mask.unsqueeze(-1).to(pa_cell_atten.device)  ## [n_patient, n_cell, 1]
        else:
            mask = None
            
        cell_output = self.attention_layer(pa_cell_atten, mask) ## [n_patient, n_hid]
        patient_output = self.res_pred_model(cell_output) ## [n_patient, 2]
        
        return patient_output
    
    
    
def response_train(model, optimizer, m, train_patients, valid_patients, test_patients, patient_info, max_cell, scheduler, device, epochs=1, patience=150):
    
    running_loss_tr = []
    running_loss_val = []
    running_loss_test = []
    running_score_tr = []
    running_score_val = []
    running_score_test = []
    
    ## Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    #for epoch in tqdm(range(epochs), desc="Epoch", leave=False):
    for epoch in range(epochs):
        
        model.train()
        
        ## Training dataset
        optimizer.zero_grad()
        tr_pa_pred = torch.zeros(len(train_patients), 2)
        tr_pa_GT = torch.zeros(len(train_patients))
        tr_pa_cell_num = torch.zeros(len(train_patients))
        for i in range(len(train_patients)):
            pa_id = str(train_patients[i])
            pa_idx = m.obs.loc[m.obs['patient'] == pa_id].index.tolist()
            tr_pa_cell_num[i] = len(pa_idx)
            tr_pa_GT[i] = 1 if patient_info[patient_info['Patient_id'] == pa_id].Response.values[0] == 'R' else 0
            
            tr_pa_pred[i] = model.forward(m, pa_idx, device)
        
        loss = nn.CrossEntropyLoss()(tr_pa_pred, tr_pa_GT.long())
        running_loss_tr.append(loss.item())
        
        _, tr_pa_pred_results = torch.max(tr_pa_pred.detach().cpu(), 1)
        response_score = f1_score(tr_pa_GT.numpy(), tr_pa_pred_results.numpy(), average='micro')
        running_score_tr.append(response_score)

        loss.backward()
        optimizer.step()
        
        #print(f'*****************Epoch [{epoch+1}/{epochs}]*****************')
        #print(f'Train Loss: {loss/10:.4f}, f1: {response_score}')
        
        model.eval()
        
        ## Val dataset
        val_pa_pred = torch.zeros(len(valid_patients), 2)
        val_pa_GT = torch.zeros(len(valid_patients))
        val_pa_cell_num = torch.zeros(len(valid_patients))
        for i in range(len(valid_patients)):
            pa_id = str(valid_patients[i])
            pa_idx = m.obs.loc[m.obs['patient'] == pa_id].index.tolist()
            val_pa_cell_num[i] = len(pa_idx)
            val_pa_GT[i] = 1 if patient_info[patient_info['Patient_id'] == pa_id].Response.values[0] == 'R' else 0
            
            val_pa_pred[i] = model.forward(m, pa_idx, device)
        
        val_loss = nn.CrossEntropyLoss()(val_pa_pred, val_pa_GT.long())
        running_loss_val.append(val_loss.item())
        
        _, val_pa_pred_results = torch.max(val_pa_pred.detach().cpu(), 1)
        response_score = f1_score(val_pa_GT.numpy(), val_pa_pred_results.numpy(), average='micro')
        running_score_val.append(response_score)
        
        scheduler.step(val_loss)

        #print(f'Val Loss: {loss/10:.4f}, f1: {response_score}')
        
        ## Test dataset
        test_pa_pred = torch.zeros(len(test_patients), 2)
        test_pa_GT = torch.zeros(len(test_patients))
        test_pa_cell_num = torch.zeros(len(test_patients))
        for i in range(len(test_patients)):
            pa_id = str(test_patients[i])
            pa_idx = m.obs.loc[m.obs['patient'] == pa_id].index.tolist()
            test_pa_cell_num[i] = len(pa_idx)
            test_pa_GT[i] = 1 if patient_info[patient_info['Patient_id'] == pa_id].Response.values[0] == 'R' else 0
            
            test_pa_pred[i] = model.forward(m, pa_idx, device)
        
        loss = nn.CrossEntropyLoss()(test_pa_pred, test_pa_GT.long())
        running_loss_test.append(loss.item())
        
        _, test_pa_pred_results = torch.max(test_pa_pred.detach().cpu(), 1)
        response_score = f1_score(test_pa_GT.numpy(), test_pa_pred_results.numpy(), average='micro')
        running_score_test.append(response_score)
        
        #print(f'Test Loss: {loss/10:.4f}, f1: {response_score}')
        
        ## Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_test_pa_pred = test_pa_pred.clone().detach()
            best_test_score = running_score_test.copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            #print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return (running_loss_tr, running_loss_val, running_loss_test, running_score_tr, running_score_val, running_score_test,
            tr_pa_pred.detach(), val_pa_pred.detach(), test_pa_pred.detach(), best_test_pa_pred, tr_pa_GT.detach(), val_pa_GT.detach(), test_pa_GT.detach(), epoch, copy.deepcopy(model.state_dict()))