import torch
import sys
import pickle as pkl
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

model_type = sys.argv[1]
dataset = sys.argv[2]

if model_type == 'vanilla': 
    model = torch.load('models/j.pth', weights_only=False)
    ModelXtoCtoY_layer = torch.load('models/xtoc.pth', weights_only=False)
    ModelXtoCtoY_layer.eval()
    model.eval()

if dataset == 'dili':
    with open(f'models/val_data_dili_vanilla.pkl', 'rb') as f:
        val_loader = pkl.load(f)
    
    predictions = np.array([])
    true_labels = np.array([])
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
        print(f'correct label {label[0].item()}\n')
        print(f'predicted label {contributions[0].sum()}\n')
        predictions = np.append(predictions, outputs[0].detach().cpu().numpy())
        true_labels = np.append(true_labels, label.bool().detach().cpu().numpy())

        contributions_dict = {}
        for i, feature in enumerate(i[0] for i in features):
            contributions_dict[feature] = contributions[0][i]
        
        contributions_dict = sorted(contributions_dict.items(), key=lambda item: item[1], reverse=True)
        print(f'contributions: {contributions_dict}')