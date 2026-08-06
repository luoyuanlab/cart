import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict
from typing import Union
from sklearn.metrics import f1_score
import copy

from .layers import AttentionLayer



class ResPredLayer(nn.Module):
    def __init__(self, sg_hid, fc_dropout=0.2):
        super(ResPredLayer, self).__init__()
        self.cl_response_fc = nn.Sequential(OrderedDict([
                                          ('linear1', nn.Linear(sg_hid, sg_hid)),
                                          ('activate1', nn.LeakyReLU()),
                                          ('layernorm1', nn.LayerNorm(sg_hid)),
                                          ('Drop1', nn.Dropout(fc_dropout)),
                                          ('linear2', nn.Linear(sg_hid, sg_hid)),
                                          ('activate2', nn.LeakyReLU()),
                                          ('layernorm2', nn.LayerNorm(sg_hid)),
                                          ('Drop2', nn.Dropout(fc_dropout)),
                                          ('linear3', nn.Linear(sg_hid, 2)),
                                         ]))
    def forward(self, x):
        x = self.cl_response_fc(x)
        return x



class PatientResponseModel(nn.Module):
    def __init__(self, input_dim, max_cell, device='cpu', fc_dropout=0, seed=None):
        super(PatientResponseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed) 
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) 
            np.random.seed(seed)  
            random.seed(seed) 
            
        self.attention_layer = AttentionLayer(input_dim).to(device)  
        self.res_pred_model = ResPredLayer(input_dim, fc_dropout=fc_dropout).to(device) 
        
        self.max_cell = max_cell
        self.device = device
        
    def to(self, device):
        self.attention_layer.to(device)
        self.res_pred_model.to(device)
        return super(PatientResponseModel, self).to(device)

    def forward(self, m, pa_idx, input_option: Union["embed", "embed_annotation"] = "embed"):

        cell_num = len(pa_idx)
        
        if input_option == "embed":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1]).to(self.device)
            pa_cell_atten[:1, :cell_num, :] = torch.tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed'], device=self.device).unsqueeze(0)
        elif input_option == "embed_annotation":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1] + m.obsm['type_prob'].shape[1]).to(self.device)
            pa_cell_atten[:1, :cell_num, :] = torch.cat([torch.Tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed'], device=self.device),
                             torch.Tensor(m[np.array(pa_idx).tolist()].obsm['type_prob'], device=self.device)
                            ], dim=1).unsqueeze(0).to(self.device)
            
        if cell_num != None:
            mask = torch.arange(pa_cell_atten.size(1)).expand(pa_cell_atten.size(0), -1).to(self.device) < cell_num
            mask = mask.unsqueeze(-1).to(pa_cell_atten.device) 
        else:
            mask = None
            
        cell_output, atten_weights = self.attention_layer(pa_cell_atten, mask) 
        patient_output = self.res_pred_model(cell_output) 
        
        return patient_output, atten_weights
    
    
    
def response_train(model, optimizer, m, train_patients, valid_patients, test_patients, max_cell, scheduler, device, epochs=1, patience=150, response_weights=[1.0, 1.0]):
    
    model = model.to(device)
    
    running_loss_tr = []
    running_loss_val = []
    running_loss_test = []
    running_score_tr = []
    running_score_val = []
    running_score_test = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    patient_response_dict = {}
    patient_cell_idx_dict = {}
    for i in range(len(train_patients)):
        pa_id = str(train_patients[i])
        patient_cell_idx_dict[pa_id] = m.obs.loc[(m.obs['patient'] == pa_id) & (m.obs['response_pick'] == 'pick')].index.tolist()
        patient_response_dict[pa_id] = 1 if m.obs[m.obs['patient'] == pa_id].Response.values[0] == 'R' else 0
    for i in range(len(valid_patients)):
        pa_id = str(valid_patients[i])
        patient_cell_idx_dict[pa_id] = m.obs.loc[(m.obs['patient'] == pa_id) & (m.obs['response_pick'] == 'pick')].index.tolist()
        patient_response_dict[pa_id] = 1 if m.obs[m.obs['patient'] == pa_id].Response.values[0] == 'R' else 0
    for i in range(len(test_patients)):
        pa_id = str(test_patients[i])
        patient_cell_idx_dict[pa_id] = m.obs.loc[(m.obs['patient'] == pa_id) & (m.obs['response_pick'] == 'pick')].index.tolist()
        patient_response_dict[pa_id] = 1 if m.obs[m.obs['patient'] == pa_id].Response.values[0] == 'R' else 0
    
    for epoch in range(epochs):
        
        model.train()
        
        optimizer.zero_grad()
        tr_pa_pred = torch.zeros(len(train_patients), 2, device=device)
        tr_pa_cell_weight = torch.zeros(len(train_patients), max_cell, device=device)
        tr_pa_GT = torch.zeros(len(train_patients), device=device)
        tr_pa_cell_num = torch.zeros(len(train_patients), device=device)
        for i in range(len(train_patients)):
            pa_id = str(train_patients[i])
            pa_idx = patient_cell_idx_dict[pa_id]
            tr_pa_cell_num[i] = len(pa_idx)
            tr_pa_GT[i] = patient_response_dict[pa_id]
            tr_pa_pred[i], tr_pa_cell_weight[i] = model.forward(m, pa_idx)
        
        loss = nn.CrossEntropyLoss(weight=torch.tensor(response_weights, device=device))(tr_pa_pred, tr_pa_GT.long())
        running_loss_tr.append(loss.item())
        
        _, tr_pa_pred_results = torch.max(tr_pa_pred.detach().cpu(), 1)
        response_score = f1_score(tr_pa_GT.cpu().numpy(), tr_pa_pred_results.cpu().numpy())
        running_score_tr.append(response_score)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        model.eval()
        
        val_pa_pred = torch.zeros(len(valid_patients), 2, device=device)
        val_pa_cell_weight = torch.zeros(len(valid_patients), max_cell, device=device)
        val_pa_GT = torch.zeros(len(valid_patients), device=device)
        val_pa_cell_num = torch.zeros(len(valid_patients), device=device)
        for i in range(len(valid_patients)):
            pa_id = str(valid_patients[i])
            pa_idx = patient_cell_idx_dict[pa_id]
            val_pa_cell_num[i] = len(pa_idx)
            val_pa_GT[i] = patient_response_dict[pa_id]
            val_pa_pred[i], val_pa_cell_weight[i] = model.forward(m, pa_idx)
        
        val_loss = nn.CrossEntropyLoss(weight=torch.tensor(response_weights, device=device))(val_pa_pred, val_pa_GT.long())
        running_loss_val.append(val_loss.item())
        
        _, val_pa_pred_results = torch.max(val_pa_pred.detach().cpu(), 1)
        response_score = f1_score(val_pa_GT.cpu().numpy(), val_pa_pred_results.cpu().numpy())
        running_score_val.append(response_score)
        
        scheduler.step(val_loss)
        torch.cuda.empty_cache()

        test_pa_pred = torch.zeros(len(test_patients), 2, device=device)
        test_pa_cell_weight = torch.zeros(len(test_patients), max_cell, device=device)
        test_pa_GT = torch.zeros(len(test_patients), device=device)
        test_pa_cell_num = torch.zeros(len(test_patients), device=device)
        for i in range(len(test_patients)):
            pa_id = str(test_patients[i])
            pa_idx = patient_cell_idx_dict[pa_id]
            test_pa_cell_num[i] = len(pa_idx)
            test_pa_GT[i] = patient_response_dict[pa_id]
            test_pa_pred[i], test_pa_cell_weight[i] = model.forward(m, pa_idx)
        
        loss = nn.CrossEntropyLoss(weight=torch.tensor(response_weights, device=device))(test_pa_pred, test_pa_GT.long())
        running_loss_test.append(loss.item())
        
        _, test_pa_pred_results = torch.max(test_pa_pred.detach().cpu(), 1)
        response_score = f1_score(test_pa_GT.cpu().numpy(), test_pa_pred_results.cpu().numpy())
        running_score_test.append(response_score)
        torch.cuda.empty_cache()

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break

    return (tr_pa_pred.detach().cpu(), val_pa_pred.detach().cpu(), test_pa_pred.detach().cpu(), tr_pa_GT.detach().cpu(), val_pa_GT.detach().cpu(), test_pa_GT.detach().cpu(), 
            tr_pa_cell_weight.detach().cpu(), val_pa_cell_weight.detach().cpu(), test_pa_cell_weight.detach().cpu(), epoch)