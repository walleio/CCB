import torch
import sys
import pickle as pkl
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

model_type = sys.argv[1]
dataset = sys.argv[2]

if model_type == 'vanilla': 
    model = torch.load('models/joint_4.pth', weights_only=False)
    ModelXtoCtoY_layer = torch.load('models/ModelXtoCtoY_layer_joint_4.pth', weights_only=False)
    ModelXtoCtoY_layer.eval()
    model.eval()
elif model_type == 'gnn':
    model = torch.load('models/gnn.pth', weights_only=False)
    
    model.eval()

if dataset == 'dili':
    with open(f'models/val_data_dili_vanilla.pkl_4', 'rb') as f:
        val_loader = pkl.load(f)
    
    predictions = np.array([])
    true_labels = np.array([])
    contributions_dict = {}
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        concept_labels = batch['concept_labels']
        features = batch['features']

        outputs = model(input_ids=input_ids.to('cuda:0').squeeze(), attention_mask=attention_mask.to('cuda:0').squeeze(), output_hidden_states=True)

        pooled_output = outputs.hidden_states[-1][:,0] 

        outputs = ModelXtoCtoY_layer(pooled_output)
        concepts = torch.stack(outputs[1:], dim=1)

        last_layer = None
        for name, m in ModelXtoCtoY_layer.named_modules():
            if name == 'sec_model':
                last_layer = m

        for name, m in last_layer.named_modules():
            if name == 'linear':
                last_layer = m

        W = last_layer.weight.squeeze()
        b = last_layer.bias.squeeze()

        contributions = concepts.squeeze().detach().cpu().numpy()*W.detach().cpu().numpy()
        predictions = np.append(predictions, outputs[0].detach().cpu().numpy())
        true_labels = np.append(true_labels, label.bool().detach().cpu().numpy())

        for i, feature in enumerate(i[0] for i in features):
            try:
                contributions_dict[feature] = (contributions.T)[i]
            except:
                contributions_dict[feature] = np.append(contributions_dict[feature], (contributions.T)[i])
    
    average_contributions = {}
    for feature in contributions_dict.keys():
        average_contributions[feature] = np.mean(abs(contributions_dict[feature]))
    
    print(average_contributions)
    print(predictions)
    print(true_labels)