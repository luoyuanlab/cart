import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict
from numbers import Integral
from typing import Mapping, Optional, Sequence, Union
from sklearn.metrics import f1_score
import copy

from .layers import AttentionLayer



RELAPSE_LABEL_MAP = {
    "No": 0,   # no relapse
    "RL": 1,   # relapse
}
RELAPSE_CLASSES = tuple(RELAPSE_LABEL_MAP)
RELAPSE_TO_INDEX = RELAPSE_LABEL_MAP
INDEX_TO_RELAPSE = {
    index: label for label, index in RELAPSE_TO_INDEX.items()
}
NUM_RELAPSE_CLASSES = len(RELAPSE_CLASSES)



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
                                          ('linear3', nn.Linear(sg_hid, NUM_RELAPSE_CLASSES)),
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
        self.device = torch.device(device)
        self.attention_layer.to(device)
        self.res_pred_model.to(device)
        return super(PatientResponseModel, self).to(device)

    def forward(self, m, pa_idx, input_option: Union["embed", "embed_annotation"] = "embed"):

        cell_num = len(pa_idx)
        
        model_dtype = next(self.attention_layer.parameters()).dtype

        if input_option == "embed":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1], dtype=model_dtype, device=self.device,)
            cell_embed = torch.as_tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed'], dtype=model_dtype, device=self.device, )
            pa_cell_atten[:1, :cell_num, :] = cell_embed.unsqueeze(0)
        elif input_option == "embed_annotation":
            pa_cell_atten = torch.zeros(1, self.max_cell, m.obsm['cell_embed'].shape[1] + m.obsm['type_prob'].shape[1], dtype=model_dtype, device=self.device,)
            cell_embed = torch.as_tensor(m[np.array(pa_idx).tolist()].obsm['cell_embed'], dtype=model_dtype, device=self.device,)
            type_prob = torch.as_tensor(m[np.array(pa_idx).tolist()].obsm['type_prob'], dtype=model_dtype, device=self.device,)
            pa_cell_atten[:1, :cell_num, :] = torch.cat([cell_embed, type_prob], dim=1).unsqueeze(0)
        else:
            raise ValueError(
                "input_option must be either 'embed' or 'embed_annotation'."
            )
        
        if cell_num != None:
            #mask = torch.arange(pa_cell_atten.size(1)).expand(pa_cell_atten.size(0), -1).to(self.device) < cell_num
            mask = torch.arange(pa_cell_atten.size(1), device=self.device).expand(pa_cell_atten.size(0), -1)< cell_num
            mask = mask.unsqueeze(-1).to(pa_cell_atten.device) 
        else:
            mask = None
            
        cell_output, atten_weights = self.attention_layer(pa_cell_atten, mask) 
        patient_output = self.res_pred_model(cell_output) 
        
        return patient_output, atten_weights
    
    
    
def response_train(
    model,
    optimizer,
    m,
    train_patients,
    valid_patients,
    test_patients,
    max_cell,
    scheduler,
    device,
    epochs=1,
    patience=150,
    response_weights: Sequence[float] = (1.0, 1.0, 1.0),
    patient_relapse_dict: Optional[Mapping[str, Union[str, int]]] = None,
    relapse_label_column: str = "Relapse",
    cell_selection_column: str = "relapse_pick",
): 

    required_obs_columns = {"patient", cell_selection_column}
    missing_obs_columns = required_obs_columns.difference(m.obs.columns)
    if missing_obs_columns:
        raise ValueError(
            "m.obs is missing required column(s): "
            + ", ".join(sorted(missing_obs_columns))
        )

    train_patients = [str(patient).strip() for patient in train_patients]
    valid_patients = [str(patient).strip() for patient in valid_patients]
    test_patients = [str(patient).strip() for patient in test_patients]

    for split_name, patients in (
        ("train", train_patients),
        ("validation", valid_patients),
        ("test", test_patients),
    ):
        if not patients:
            raise ValueError(f"The {split_name} patient split is empty.")
        if len(patients) != len(set(patients)):
            raise ValueError(
                f"The {split_name} patient split contains duplicate IDs."
            )

    train_set = set(train_patients)
    valid_set = set(valid_patients)
    test_set = set(test_patients)
    split_overlap = (
        (train_set & valid_set)
        | (train_set & test_set)
        | (valid_set & test_set)
    )
    if split_overlap:
        raise ValueError(
            "Patient IDs must not occur in more than one split; overlapping "
            f"IDs: {sorted(split_overlap)}"
        )

    class_weights = torch.as_tensor(
        response_weights,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    if class_weights.shape != (NUM_RELAPSE_CLASSES,):
        raise ValueError(
            "response_weights must contain exactly three values ordered as "
            "[No, RL]."
        )
    if (
        not torch.isfinite(class_weights).all().item()
        or (class_weights <= 0).any().item()
    ):
        raise ValueError("All response_weights must be positive and finite.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = model.to(device)
    
    running_loss_tr = []
    running_loss_val = []
    running_loss_test = []
    running_score_tr = []
    running_score_val = []
    running_score_test = []
    
    best_val_loss = float('inf')
    best_epoch = None
    best_model_state = None
    patience_counter = 0
    
    patient_relapse_index_dict = {}
    patient_cell_idx_dict = {}
    
    def relapse_value_to_index(relapse_value, patient_id):
        if (
            isinstance(relapse_value, Integral)
            and not isinstance(relapse_value, (bool, np.bool_))
        ):
            relapse_index = int(relapse_value)
            if relapse_index not in INDEX_TO_RELAPSE:
                raise ValueError(
                    f"Unknown relapse class index {relapse_index!r} for "
                    f"patient {patient_id!r}; expected one of "
                    f"{list(INDEX_TO_RELAPSE)}."
                )
            return relapse_index

        relapse_label = str(relapse_value).strip()
        if relapse_label not in RELAPSE_TO_INDEX:
            raise ValueError(
                f"Unknown relapse label {relapse_label!r} for patient "
                f"{patient_id!r}; expected one of {list(RELAPSE_CLASSES)}."
            )
        return RELAPSE_TO_INDEX[relapse_label]

    normalized_relapse_dict = None
    if patient_relapse_dict is not None:
        normalized_relapse_dict = {}
        for patient, relapse_value in patient_relapse_dict.items():
            patient_id = str(patient).strip()
            relapse_index = relapse_value_to_index(
                relapse_value, patient_id
            )
            if (
                patient_id in normalized_relapse_dict
                and normalized_relapse_dict[patient_id] != relapse_index
            ):
                raise ValueError(
                    f"Patient {patient_id!r} has conflicting relapse labels."
                )
            normalized_relapse_dict[patient_id] = relapse_index
    elif relapse_label_column not in m.obs.columns:
        raise ValueError(
            f"m.obs does not contain {relapse_label_column!r}. Pass "
            "patient_relapse_dict={patient_id: relapse_label} to "
            "response_train()."
        )

    obs_patient_ids = m.obs["patient"].astype(str).str.strip()
    picked_cell_mask = (
        m.obs[cell_selection_column].astype(str).str.strip().eq("pick")
    )

    all_patients = train_patients + valid_patients + test_patients
    for pa_id in all_patients:
        patient_mask = obs_patient_ids.eq(pa_id)
        if not patient_mask.any():
            raise ValueError(f"Patient {pa_id!r} has no cells in m.obs.")

        selected_cells = m.obs.index[patient_mask & picked_cell_mask].tolist()
        if not selected_cells:
            raise ValueError(
                f"Patient {pa_id!r} has no cells marked 'pick'."
            )
        if len(selected_cells) > max_cell:
            raise ValueError(
                f"Patient {pa_id!r} has {len(selected_cells)} selected cells, "
                f"which exceeds max_cell={max_cell}."
            )
        patient_cell_idx_dict[pa_id] = selected_cells

        if normalized_relapse_dict is not None:
            if pa_id not in normalized_relapse_dict:
                raise ValueError(
                    f"No relapse label was provided for patient {pa_id!r}."
                )
            relapse_index = normalized_relapse_dict[pa_id]
        else:
            relapse_labels = (
                m.obs.loc[patient_mask, relapse_label_column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            relapse_labels = relapse_labels[relapse_labels.ne("")].unique()
            if len(relapse_labels) != 1:
                raise ValueError(
                    f"Patient {pa_id!r} must have exactly one relapse label; "
                    f"found {relapse_labels.tolist()}."
                )
            relapse_index = relapse_value_to_index(
                relapse_labels[0], pa_id
            )
        patient_relapse_index_dict[pa_id] = relapse_index

    missing_train_classes = sorted(
        set(range(NUM_RELAPSE_CLASSES)).difference(
            patient_relapse_index_dict[pa_id] for pa_id in train_patients
        )
    )
    if missing_train_classes:
        missing_labels = [
            INDEX_TO_RELAPSE[index] for index in missing_train_classes
        ]
        raise ValueError(
            "Every relapse class must occur in the training split; missing "
            f"class(es): {missing_labels}."
        )

    def run_patient_split(patients):
        patient_predictions = []
        patient_cell_weights = []
        patient_targets = []
        patient_cell_counts = []

        for pa_id in patients:
            pa_idx = patient_cell_idx_dict[pa_id]
            prediction, attention_weights = model.forward(m, pa_idx)
            prediction = prediction.reshape(-1)
            if prediction.shape != (NUM_RELAPSE_CLASSES,):
                raise ValueError(
                    "The model must return exactly three logits per patient; "
                    f"received shape {tuple(prediction.shape)} for {pa_id!r}."
                )

            attention_weights = attention_weights.reshape(-1)
            if attention_weights.numel() != max_cell:
                raise ValueError(
                    f"Expected {max_cell} attention weights for patient "
                    f"{pa_id!r}, received {attention_weights.numel()}."
                )

            patient_predictions.append(prediction)
            patient_cell_weights.append(attention_weights)
            patient_targets.append(patient_relapse_index_dict[pa_id])
            patient_cell_counts.append(len(pa_idx))

        return (
            torch.stack(patient_predictions),
            torch.stack(patient_cell_weights),
            torch.tensor(patient_targets, dtype=torch.long, device=device),
            torch.tensor(patient_cell_counts, dtype=torch.long, device=device),
        )

    def macro_f1(targets, predictions):
        return f1_score(
            targets.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
            labels=list(range(NUM_RELAPSE_CLASSES)),
            average="macro",
            zero_division=0,
        )
    
    for epoch in range(epochs):
        
        model.train()
        
        optimizer.zero_grad()
        
        (
            tr_pa_pred,
            tr_pa_cell_weight,
            tr_pa_GT,
            tr_pa_cell_num,
        ) = run_patient_split(train_patients)

        loss = criterion(tr_pa_pred, tr_pa_GT)
        running_loss_tr.append(loss.item())
        
        tr_pa_pred_results = tr_pa_pred.detach().argmax(dim=1)
        relapse_score = macro_f1(tr_pa_GT, tr_pa_pred_results)
        running_score_tr.append(relapse_score)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            (
                val_pa_pred,
                val_pa_cell_weight,
                val_pa_GT,
                val_pa_cell_num,
            ) = run_patient_split(valid_patients)

        val_loss = criterion(val_pa_pred, val_pa_GT)
        running_loss_val.append(val_loss.item())
        
        val_pa_pred_results = val_pa_pred.detach().argmax(dim=1)
        relapse_score = macro_f1(val_pa_GT, val_pa_pred_results)
        running_score_val.append(relapse_score)
        
        scheduler.step(val_loss.item())
        torch.cuda.empty_cache()

        with torch.no_grad():
            (
                test_pa_pred,
                test_pa_cell_weight,
                test_pa_GT,
                test_pa_cell_num,
            ) = run_patient_split(test_patients)

        test_loss = criterion(test_pa_pred, test_pa_GT)
        running_loss_test.append(test_loss.item())
        
        test_pa_pred_results = test_pa_pred.detach().argmax(dim=1)
        relapse_score = macro_f1(test_pa_GT, test_pa_pred_results)
        running_score_test.append(relapse_score)
        torch.cuda.empty_cache()

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_model_state = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in model.state_dict().items()
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break

    return (running_loss_tr, running_loss_val, running_loss_test, tr_pa_pred.detach().cpu(), val_pa_pred.detach().cpu(), test_pa_pred.detach().cpu(), tr_pa_GT.detach().cpu(), 
            val_pa_GT.detach().cpu(), test_pa_GT.detach().cpu(), tr_pa_cell_weight.detach().cpu(), val_pa_cell_weight.detach().cpu(), test_pa_cell_weight.detach().cpu(), epoch)#, copy.deepcopy(model.state_dict()))