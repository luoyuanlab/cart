import gc
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import time
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.nn.parameter import Parameter
from typing import Optional
from CART.layers import HGAT_sparse, HGNN_fc
from CART.utils import ExpValueEncoder, BatchLabelEncoder
from CART.AdversarialDiscriminator import AdversarialDiscriminator
from CART.Linformer import LinformerTransformer
from CART.layers import Gene_to_Cell_AttentionLayer
import CART.hg_ops as hgo

class pretrain_tf(nn.Module):
    def __init__(self, 
                 H, 
                 type_yuniques,
                 data_yuniques,
                 n_input_gene, ## nodes dimension
                 n_hid,  
                 combined_type: str = 'sum',
                 dropout=0.5, 
                 fn=None, 
                 seed=0, 
                 atype='additive', 
                 metric='f1', 
                 w_loss1=1,
                 w_loss2=1,
                 w_loss3=1,
                 fc_dropout=0.5, 
                 use_data_batch=False,
                 jk=False):
        super(pretrain_tf, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
        self.w_loss1 = w_loss1
        self.w_loss2 = w_loss2
        self.w_loss3 = w_loss3
        
        HT = H.T ## HT: [edges, nodes]
        self.e_degs = H.sum(0) ## egde degree
        self.n_degs = H.sum(1) ## node degree

        self.HTa = HT / HT.sum(1, keepdim=True) ## normalize?; HTa: [edges, nodes]
            
        self.pair = HT.nonzero(as_tuple=False).t() ## [2, non_zero_numbers]. nonzero: returns a 2-D tensor where each row is the index for a nonzero value.
        
        self.fn = fn
        self.yuniques = type_yuniques
        n_class = len(type_yuniques)
        n_class1 = len(data_yuniques)

        self.combined_type = combined_type
        if combined_type not in ["sum", "multiplicative"]:
            raise ValueError(f"Unknown combined_type: {combined_type}, please select 'sum' or 'multiplicative'")
        self.use_data_batch = use_data_batch

        ## HGAT layer, for Gene embedding
        self.hgc1 = HGAT_sparse(n_input_gene, n_hid, dropout=dropout, alpha=0.2, transfer=True, bias=True, concat=False) ## HGAT: Heterogeneous Graph Attention Networks
        self.hgc11_norm = nn.LayerNorm(n_hid)
        self.hgc12_norm = nn.LayerNorm(n_hid)
        #self.hgc11_norm = nn.BatchNorm1d(n_hid)
        #self.hgc12_norm = nn.BatchNorm1d(n_hid)
        self.hgc2 = HGAT_sparse(n_hid, n_hid, dropout=dropout, alpha=0.2, transfer=True, bias=True, concat=False)
        
        ## Gene exp embedding
        self.exp_embed = ExpValueEncoder(embedding_dim=n_hid, dropout=dropout, max_value=np.inf)  ## [batch, genes, n_hid]
        ## Norm embedding after combine exp and Gene embedding
        self.bn = nn.BatchNorm1d(n_hid, eps=1e-5)  ## in nn.BatchNorm1d, the value should be the 2nd dimension when 3D matrix as the input, so here is n_hid after permute(0,2,1)
        
        self.jk = jk 
        sg_hid = n_hid *2 if self.jk else n_hid

        ## transformer encoder
        self.tfe = LinformerTransformer(num_layers=6, dim=n_hid, seq_len=n_input_gene, heads=4, k=256)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
            self.tfe = nn.DataParallel(self.tfe)

        ## Data batch Encoder
        if self.use_data_batch:
            self.data_batch_encoder = BatchLabelEncoder(n_class1, n_hid)
        sg_jk_hid = sg_hid *2 if use_data_batch else sg_hid

        #self.gene_to_cell_fc = Gene_to_Cell_AttentionLayer(in_feat=n_hid)

        
        self.cl_type_fc = nn.Sequential(OrderedDict([
                                                  #('linear1', HGNN_fc(sg_jk_hid, sg_hid)),
                                                  ('linear1', HGNN_fc(sg_hid, sg_hid)),
                                                  ('activate1', nn.LeakyReLU()),
                                                  ('layernorm1', nn.LayerNorm(sg_hid)),
                                                  ('Drop1', nn.Dropout(fc_dropout)),
                                                  ('linear2', HGNN_fc(sg_hid, sg_hid)),
                                                  ('activate2', nn.LeakyReLU()),
                                                  ('layernorm2', nn.LayerNorm(sg_hid)),
                                                  ('Drop2', nn.Dropout(fc_dropout)),
                                                  ('linear3', HGNN_fc(sg_hid, n_class)),
                                                 ]))
        self.gene_exp_fc = nn.Sequential(OrderedDict([
                                                  ('linear1', HGNN_fc(sg_jk_hid, sg_hid)),
                                                  ('activate1', nn.LeakyReLU()),
                                                  ('Drop1', nn.Dropout(fc_dropout)),
                                                  ('linear2', HGNN_fc(sg_hid, sg_hid)),
                                                  ('activate2', nn.LeakyReLU()),
                                                  ('Drop2', nn.Dropout(fc_dropout)),
                                                  ('linear3', HGNN_fc(sg_hid, 1)),
                                                 ]))
        
        
        
        ## learn cell batch Adversarial
        self.grad_reverse_discriminator = AdversarialDiscriminator(
                sg_hid,
                n_cls=len(data_yuniques),
                reverse_grad=True,
            )


        self.metric = metric
        
        self.report = defaultdict(list)
        
        self.a = nn.Parameter(torch.zeros(size=(n_hid, 1)))   
        self.a2 = nn.Parameter(torch.zeros(size=(n_hid, 1)))
        self.class_token = nn.Parameter(torch.zeros(1, 1, n_hid))
        
        self.n_hid = n_hid
        
        stdv = 1. / math.sqrt(n_hid)
        self.a.data.uniform_(-stdv, stdv)
        self.a2.data.uniform_(-stdv, stdv)
        nn.init.normal_(self.class_token, std=0.2)
        
    def to(self, device):
        self.pair = self.pair.to(device)
        
        self.HTa = self.HTa.to(device)
        
        return super(pretrain_tf, self).to(device)
    
    def _check_data_batch_labels(self, data_batch_ids: Tensor) -> None:
        if self.use_data_batch:
            assert data_batch_ids is not None
        elif data_batch_ids is not None:
            raise ValueError(
                "data_batch_ids should only be provided when `self.use_data_batch` is True"
            )

    ## x: [nodes, nodes]; xe: [edges, nodes]; sgs: [patients/cells (train/val), nodes]
    def encode(self, x, xe, sgs): 
        
        ## x: [nodes, nodes]; xe: [edges, nodes]; pair: [2, non_zero_numbers], sparse expression of HT [edge, node]; a: data.uniform_(-stdv, stdv) [n_hid, 1]
        x1, xe = self.hgc1(x, xe, self.pair, self.a)  ## HGAT: Heterogeneous Graph Attention Networks, [n_node, out_ch(n_hid)], [n_edge, out_ch(n_hid)]
        x1 = self.hgc11_norm(x1)
        xe = self.hgc12_norm(xe)
        
        ## x: [nodes, n_hid]; xe: [edges, n_hid]; pair: [2, non_zero_numbers], sparse expression of HT [edge, node]; a2: data.uniform_(-stdv, stdv) [n_hid, 1]
        xn, xe = self.hgc2(x1, xe, self.pair, self.a2)  ## HGAT: Heterogeneous Graph Attention Networks, [n_node, out_ch(n_hid)], [n_edge, out_ch(n_hid)]
        
        if self.jk:
            xn = torch.cat((xn, x1), 1)  ## [n_node/genes, 2*n_hid]
        
        sgs_exp = self.exp_embed(sgs)  ## [cells, genes, n_hid]
        
        if self.combined_type == 'sum':
            sgs = xn + sgs_exp ## [cells, genes, n_hid]
        else:
            sgs = xn * sgs_exp ## equal to torch.mul(aaa, bbb)??, [cells, genes, n_hid]
        '''
        sgs_exp_mean = sgs_exp.mean(axis=1).unsqueeze(1)  ## [cells, 1, n_hid]
        if self.combined_type == 'sum':
            sgs_cell = self.class_token.expand(sgs_exp_mean.size(0), -1, -1) + sgs_exp_mean
        else:
            sgs_cell = self.class_token.expand(sgs_exp_mean.size(0), -1, -1) * sgs_exp_mean  
        #sgs = torch.cat([sgs_cell, sgs], dim=1)  ## (batch, seq_len+1, embsize)
        '''    
        
        #sgs = torch.flatten(sgs.contiguous(), start_dim=1) 
        sgs = sgs.contiguous().transpose(1,2)
        try:
            sgs = self.bn(sgs)
        except:
            sgs = self.bn.to('cpu')(sgs.to('cpu')).to(device)

        sgs = sgs.contiguous().transpose(1,2)
        
        sgs = self.tfe(sgs)

        return sgs, xn, xe  ## [cells, genes+1, n_hid], [n_node, n_hid], [n_edge, n_hid]
    
    
    
    def forward_pretrain(self, x_nfts, xe, sgs, data_batch_ids):
        
        x_gene, xno, xeo = self.encode(x_nfts, ## [nodes, nodes]
                                  xe, ## [edges, nodes]
                                  sgs) ## [patients (train/val), nodes]
        ## x_gene, xno, xeo: [cells, genes, n_hid], [n_node, n_hid], [n_edge, n_hid]

        #x_cell = self.gene_to_cell_fc(x_gene.contiguous().transpose(1,2)).squeeze()
        #x_cell = self.gene_to_cell_fc(x_gene).squeeze()
        x_cell = x_gene.mean(axis=1).squeeze()
        
        batch_prob = self.grad_reverse_discriminator(x_cell)
        
        if self.use_data_batch:
            batch_emb = self.data_batch_encoder(data_batch_ids)  ## (batch, embsize)
            x_gene = torch.cat([x_gene, batch_emb.unsqueeze(1).repeat(1, x_gene.shape[1], 1) ## [batch, token, emb]
                           ], dim=2)

        type_prob = self.cl_type_fc(x_cell)            

        gene_preds = self.gene_exp_fc(x_gene).squeeze()
        
        return type_prob, gene_preds, batch_prob, x_gene, x_cell, xno, xeo
        

    
    def fit_pretrain(self, x_nfts, ## x(nfts): node feature, [nodes(genes), nodes(genes)], can be one hot matrix as the initial input
                     sgs, ## sgs(m): input features [patients/cells, nodes(genes)]
                     dataloader_train_idx, 
                     dataloader_val_idx, 
                     dataloader_test_idx,
                     y, type_cls_loss, batch_cls_loss, exp_pred_loss, optimizer, scheduler, device, num_epochs=25, print_freq=1,
                     update_feature: Union['loss', 'f1', 'acc'] = 'loss',
                     data_batch_ids: Optional[Tensor] = None,  ## (batch,)
                     exp_loss_weights: Optional[Tensor] = None,  ## (batch, genes)
                    ):
        
        try:
            self._check_data_batch_labels(data_batch_ids)
        except:
            import warnings

            warnings.warn(
                "batch_ids is required but not provided, using zeros instead"
            )
            data_batch_ids = torch.zeros(
                sgs.shape[0], dtype=torch.long, device=sgs.device
            )

        since = time.time()
        
        if exp_loss_weights == None:
            exp_loss_weights = torch.ones_like(sgs).to(device)
        
        best_model_wts = copy.deepcopy(self.state_dict())
        
        if update_feature in ['f1', 'acc']:
            compare_score = 0.0
        else:
            compare_score = -np.inf

        '''
        torch.mm of * - performs a matrix multiplication without broadcasting - (2D tensor) by (2D tensor)
        torch.mul - performs a elementwise multiplication with broadcasting - (Tensor) by (Tensor or Number)
        torch.matmul - matrix product with broadcasting - (Tensor) by (Tensor) with different behaviors depending on the tensor shapes (dot product, matrix product, batched matrix products).
        '''
        xe = self.HTa.mm(x_nfts) ## [edges, nodes] * [nodes, nodes] = [edges, nodes] 

        loss_list = []
        for epoch in range(num_epochs):
            if epoch % print_freq == 0:
                print('-' * 20)
                print(f'Epoch {epoch}/{num_epochs - 1}')
            
            ## training step
            self.train()
            i = 0
            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            y_phase_pred = torch.Tensor().to(device)
            y_phase = torch.Tensor().to(device)
            #y1_total = torch.Tensor()
            for idx in dataloader_train_idx:
                i += 1
                if i%100 == 0:
                    print("{}".format(i), end=" ")
                
                try:
                    sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X.todense()).to(device)
                except:
                    sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X).to(device)

                sub_data_batch_ids = data_batch_ids[idx]
                idx = torch.tensor(np.array(idx, dtype=int))
                
                optimizer.zero_grad()
                
                type_prob, gene_preds, batch_prob, _, _, xno, xeo = self.forward_pretrain(x_nfts, ## [nodes, nodes]
                                                                                      xe, ## [edges, nodes]
                                                                                      sub_sgs, ## [patients (train/val), nodes]
                                                                                      sub_data_batch_ids)
                
                
                ## Create a mask for valid labels, Apply mask to output and target
                type_label_mask = (y[idx] != -1)
                if type_label_mask.sum() > 0:
                    loss1 = type_cls_loss(type_prob[type_label_mask], y[idx][type_label_mask])
                else:
                    loss1 = torch.tensor(0)
                
                loss2 = exp_pred_loss(gene_preds, sub_sgs, exp_loss_weights[idx])
                loss3 = batch_cls_loss(batch_prob, sub_data_batch_ids)
                loss = (self.w_loss1*loss1 + self.w_loss2*loss2 + self.w_loss3*loss3) / (self.w_loss1 + self.w_loss2 + self.w_loss3)
                
                if i%500 == 0:
                    print(loss1.item(), loss2.item(), loss3.item(), end=" ")
                running_loss1 += loss1
                running_loss2 += loss2
                running_loss3 += loss3
                running_loss += loss
                
                _, type_preds = torch.max(type_prob, 1)
                y_phase_pred = torch.cat((y_phase_pred, type_preds), 0)
                y_phase = torch.cat((y_phase, y[idx]), 0)
                
                train_xno = xno
                train_xeo = xeo

                loss.backward()
                optimizer.step()
                
                del sub_sgs, idx, type_prob, batch_prob, gene_preds, type_preds, xno, xeo, type_label_mask
                gc.collect()
                torch.cuda.empty_cache()
            
            y_phase_pred = y_phase_pred.detach().cpu().numpy()
            y_phase = y_phase.detach().cpu().numpy()
            type_label_mask = (y_phase != -1)
            epoch_cm = confusion_matrix(y_phase[type_label_mask], y_phase_pred[type_label_mask])

            epoch_loss1 = running_loss1.item() / i
            epoch_loss2 = running_loss2.item() / i
            epoch_loss3 = running_loss3.item() / i
            epoch_loss = running_loss.item() / i
            if self.metric == 'f1':
                epoch_score = f1_score(y_phase[type_label_mask], y_phase_pred[type_label_mask], average='micro')
            elif self.metric == 'acc':
                epoch_score = accuracy_score(y_phase[type_label_mask], y_phase_pred[type_label_mask])
            else:
                sys.exit(f'unsupported metric {self.metric}')
                    
            self.report['epoch'].append(epoch)
            self.report['train_loss'].append(epoch_loss)
            self.report['train_loss1'].append(epoch_loss1)
            self.report['train_loss2'].append(epoch_loss2)
            self.report['train_loss3'].append(epoch_loss3)
            self.report['train_score'].append(epoch_score)
            y_tr_pred, y_tr = y_phase_pred, y_phase
            train_cm = epoch_cm
            train_loss = epoch_loss
            train_score = epoch_score
                
            if epoch % print_freq == 0:
                print(f'Train Loss: {epoch_loss:.4f} {epoch_loss1:.4f} {epoch_loss2:.4f} {epoch_loss3:.4f} {self.metric}: {epoch_score:.4f}')
                
            del y_phase_pred, y_phase, type_label_mask
            gc.collect()
            torch.cuda.empty_cache()

            ## validation step
            self.eval()
            with torch.no_grad():
                i = 0
                running_loss = 0.0
                running_loss1 = 0.0
                running_loss2 = 0.0
                running_loss3 = 0.0
                y_phase_pred = torch.Tensor().to(device)
                y_phase = torch.Tensor().to(device)
                for idx in dataloader_val_idx:
                    i += 1
                    if i%100 == 0:
                        print("{}".format(i),end=" ")

                    try:
                        sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X.todense()).to(device)
                    except:
                        sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X).to(device)
                    idx = torch.tensor(np.array(idx, dtype=int))
                    sub_data_batch_ids = data_batch_ids[idx]
                    
                    type_prob, gene_preds, batch_prob, _, _, xno, xeo = self.forward_pretrain(x_nfts, ## [nodes, nodes]
                                                                                          xe, ## [edges, nodes]
                                                                                          sub_sgs, ## [patients (train/val), nodes]
                                                                                          sub_data_batch_ids)

                    ## Create a mask for valid labels, Apply mask to output and target
                    type_label_mask = (y[idx] != -1)
                    if type_label_mask.sum() > 0:
                        loss1 = type_cls_loss(type_prob[type_label_mask], y[idx][type_label_mask])
                    else:
                        loss1 = torch.tensor(0)

                    loss2 = exp_pred_loss(gene_preds, sub_sgs, exp_loss_weights[idx])
                    loss3 = batch_cls_loss(batch_prob, sub_data_batch_ids)
                    loss = (self.w_loss1*loss1 + self.w_loss2*loss2 + self.w_loss3*loss3) / (self.w_loss1 + self.w_loss2 + self.w_loss3)

                    if i%500 == 0:
                        print(loss1.item(), loss2.item(), loss3.item(), end=" ")
                    running_loss1 += loss1
                    running_loss2 += loss2
                    running_loss3 += loss3
                    running_loss += loss

                    _, type_preds = torch.max(type_prob, 1)
                    y_phase_pred = torch.cat((y_phase_pred, type_preds), 0)
                    y_phase = torch.cat((y_phase, y[idx]), 0)

                    del sub_sgs, idx, type_prob, batch_prob, gene_preds, type_preds, xno, xeo, type_label_mask
                    gc.collect()
                    torch.cuda.empty_cache()

                epoch_loss1 = running_loss1.item() / i
                epoch_loss2 = running_loss2.item() / i
                epoch_loss3 = running_loss3.item() / i
                epoch_loss = running_loss.item() / i
                y_phase_pred = y_phase_pred.detach().cpu().numpy()
                y_phase = y_phase.detach().cpu().numpy()
                type_label_mask = (y_phase != -1)
                epoch_cm = confusion_matrix(y_phase[type_label_mask], y_phase_pred[type_label_mask])
                
                if self.metric == 'f1':
                    epoch_score = f1_score(y_phase[type_label_mask], y_phase_pred[type_label_mask], average='micro')
                elif self.metric == 'acc':
                    epoch_score = accuracy_score(y_phase[type_label_mask], y_phase_pred[type_label_mask])
                else:
                    sys.exit(f'unsupported metric {self.metric}')

                self.report['val_loss'].append(epoch_loss)
                self.report['val_loss1'].append(epoch_loss1)
                self.report['val_loss2'].append(epoch_loss2)
                self.report['val_loss3'].append(epoch_loss3)
                self.report['val_score'].append(epoch_score)

                scheduler.step(epoch_loss)

                if epoch % print_freq == 0:
                    print(f'Valid Loss: {epoch_loss:.4f} {epoch_loss1:.4f} {epoch_loss2:.4f} {epoch_loss3:.4f} {self.metric}: {epoch_score:.4f}')

                if update_feature in ['f1', 'acc']:
                    update_score = epoch_score
                else:
                    update_score = -epoch_loss
                    
                if update_score > compare_score:
                    print(f'updata_score: {update_score:.4f}; compare_score: {compare_score:.4f}; need update!')
                else:
                    print(f'updata_score: {update_score:.4f}; compare_score: {compare_score:.4f}; not update!')
                
                if update_score > compare_score:
                    if update_feature in ['f1', 'acc']:
                        compare_score = epoch_score
                    else:
                        compare_score = -epoch_loss
                    best_epoch = epoch
                    best_train_score = train_score
                    best_train_loss =  train_loss
                    best_train_cm = train_cm
                    best_val_score = epoch_score
                    best_val_loss = epoch_loss
                    best_val_cm = epoch_cm
                    best_model_wts = copy.deepcopy(self.state_dict())
                    best_train_xno = train_xno.detach().cpu()
                    best_train_xeo = train_xeo.detach().cpu()

                    ## test step
                    pred_total = torch.Tensor().to(device)
                    output_total = torch.Tensor().to(device)
                    y_total = torch.Tensor().to(device)
                    running_loss = 0.0
                    running_loss1 = 0.0
                    running_loss2 = 0.0
                    running_loss3 = 0.0
                    i = 0
                    for idx in dataloader_test_idx:
                        i += 1
                        if i%100 == 0:
                            print("{}".format(i),end=" ")

                        try:
                            sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X.todense()).to(device)
                        except:
                            sub_sgs = torch.Tensor(sgs[np.array(idx).tolist()].X).to(device)
                        idx = torch.tensor(np.array(idx, dtype=int))
                        
                        type_preds, type_prob, gene_preds, batch_prob, _, _ = self.predict_pretrain(x_nfts,
                                                                                                  xe, 
                                                                                                  sub_sgs,
                                                                                                  data_batch_ids[idx],)

                        ## Create a mask for valid labels, Apply mask to output and target
                        type_label_mask = (y[idx] != -1)
                        if type_label_mask.sum() > 0:
                            loss1 = type_cls_loss(type_prob[type_label_mask], y[idx][type_label_mask])
                        else:
                            loss1 = torch.tensor(0)
                        
                        loss2 = exp_pred_loss(gene_preds, sub_sgs, exp_loss_weights[idx])
                        loss3 = batch_cls_loss(batch_prob, data_batch_ids[idx])
                        loss = (self.w_loss1*loss1 + self.w_loss2*loss2 + self.w_loss3*loss3) / (self.w_loss1 + self.w_loss2 + self.w_loss3)

                        pred_total = torch.cat((pred_total, type_preds), 0)
                        output_total = torch.cat((output_total, type_prob), 0)
                        y_total = torch.cat((y_total, y[idx]), 0)

                        running_loss1 += loss1
                        running_loss2 += loss2
                        running_loss3 += loss3
                        running_loss += loss

                        del sub_sgs, idx, type_prob, gene_preds, type_preds, batch_prob, type_label_mask
                        gc.collect()
                        torch.cuda.empty_cache()

                    test_loss1 = running_loss1.item() / i
                    test_loss2 = running_loss2.item() / i
                    test_loss3 = running_loss3.item() / i
                    test_loss = running_loss.item() / i

                    best_y_tr_pred, best_y_tr = y_tr_pred, y_tr
                    y_val_pred, y_val = y_phase_pred, y_phase
                    y_test_pred = pred_total.detach().cpu().numpy()
                    y_test = y_total.detach().cpu().numpy()
                    type_label_mask = (y_test != -1)
                    test_cm = confusion_matrix(y_test[type_label_mask], y_test_pred[type_label_mask])

                    if self.metric == 'f1':
                        test_score = f1_score(y_test[type_label_mask], y_test_pred[type_label_mask], average='micro')
                    elif self.metric == 'acc':
                        test_score = accuracy_score(y_test[type_label_mask], y_test_pred[type_label_mask])
                    else:
                        sys.exit(f'unsupported metric {self.metric}')

                    if update_feature in ['f1', 'acc']:
                        print(f'Updating val {self.metric}: {best_val_score:4f}; test {self.metric}: {test_score:4f}')
                    else:
                         print(f'Updating val loss: {best_val_score:4f}; test loss: {test_loss:4f}; test {self.metric}: {test_score:4f}')

                    del pred_total, output_total, y_total, y_phase_pred, y_phase, type_label_mask
                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    del y_phase_pred, y_phase, type_label_mask
                    gc.collect()
                    torch.cuda.empty_cache()

            time_elapsed = time.time() - since
            print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if self.fn is not None:
            torch.save({'epoch_idx': best_epoch,
                        'state_dict': self.state_dict(),
                        'best_state_dict': best_model_wts,
                        'optimizer': optimizer.state_dict(),
                        'type_yuniques': self.yuniques,
                        'best_train_score': best_train_score,
                        'best_train_loss': best_train_loss,
                        'best_train_cm': best_train_cm,
                        'best_train_xno': best_train_xno,
                        'best_train_xeo': best_train_xeo,
                        'best_val_score': best_val_score,
                        'best_val_loss': best_val_loss,
                        'best_val_cm': best_val_cm,
                        'test_score': test_score,
                        'test_loss': test_loss,
                        'test_cm': test_cm,
                        'y_test': y_test,
                        'y_test_pred': y_test_pred,
                        'y_val': y_val,
                        'y_val_pred': y_val_pred,
                        'y_tr': best_y_tr,
                        'y_tr_pred': best_y_tr_pred,
                        'xn': x_nfts,
                        'xe': xe,
                        'report': self.report,
            }, f'{self.fn}_pretrain.ckpt')

        fig = plt.figure()
        plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss_pretrain.png', bbox_inches='tight')
        plt.close()
        
        fig = plt.figure()
        plt.plot(self.report['train_score'], label=f'Train {self.metric}')
        plt.plot(self.report['val_score'], label=f'Val {self.metric}')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_{self.metric}_pretrain.png', bbox_inches='tight')
        plt.close()
        
        self.load_state_dict(best_model_wts)
        return self
    
    

    def predict_pretrain(self, x, xe, sgs, data_batch_ids, grad_need=False):   
        self.eval()  
        
        with torch.no_grad():
            type_prob, gene_preds, batch_prob, x_gene, x_cell, xno, xeo = self.forward_pretrain(x, ## [nodes, nodes]
                                                                                  xe, ## [edges, nodes]
                                                                                  sgs, ## [patients (train/val), nodes]
                                                                                  data_batch_ids)
            

            _, type_preds = torch.max(type_prob, 1)           

        return type_preds, type_prob, gene_preds, batch_prob, x_gene, x_cell


    
    def show_report(self):
        return pd.DataFrame(self.report)
