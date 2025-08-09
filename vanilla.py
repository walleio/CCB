import sys
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from backbone import ModelXtoCtoY_function
import numpy as np
import random
from sklearn.metrics import roc_auc_score

# this is the tokenizer that we use for encoding SMILES strings
model_name = 'mikemayuare/SMILY-APE-BBBP'
tokenizer = APETokenizer()
tokenizer.load_vocabulary('apetokenizer/tokenizer.json')
model = AutoModelForSequenceClassification.from_pretrained('mikemayuare/SMILY-APE-BBBP').to('cuda')

# get the data type and number of epochs from the command line
data_type = sys.argv[1]
num_epochs = int(sys.argv[2])
experiment = sys.argv[3]

# load data
DATA = {}
DATA['train'] = pd.read_csv(f'data/train_{data_type}.csv')
DATA['val'] = pd.read_csv(f'data/val_{data_type}.csv')
DATA['test'] = pd.read_csv(f'data/test_{data_type}.csv')

# get data statistics
means = DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).mean().values
stds = DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).std().values

# there are 200 features, so we randomly sample 10 of them
features = random.sample(range(200), 10)

num_concepts = len(features)

# create the dataset
class MyDataset(Dataset):
    def __init__(self, split, label_means, label_stds):
        self.data = DATA[split]
        self.means = label_means
        self.stds = label_stds
        self.stds[self.stds == 0] = 1
        self.labels = self.data['Y']
        self.text = self.data["Drug"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tokenized_text = tokenizer(
            self.text[index], 
            max_length=64,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
        )

        return_dict = {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'label': torch.tensor(self.labels[index], dtype=torch.float),
            'concept_labels': torch.tensor((self.data.drop(columns=['Drug', 'Y', 'Drug_ID']).iloc[index].values[features] - self.means[features]) / self.stds[features])
        }
        return return_dict

# create the dataloader
train_loader = DataLoader(MyDataset('train', means, stds), batch_size=32, shuffle=True)
val_loader = DataLoader(MyDataset('val', means, stds), batch_size=32, shuffle=False)
test_loader = DataLoader(MyDataset('test', means, stds), batch_size=32, shuffle=False)

# num_concepts is the number of concepts, expand_dim is the dimension of the expanded layer (0 means no expansion)
if experiment == 'baseline':
    ModelXtoCtoY_layer = torch.nn.Sequential(
        torch.nn.Linear(768, num_concepts),
        torch.nn.Linear(num_concepts, 1)
    ).to('cuda:0')
else:
    ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0).to('cuda:0')

loss_C = torch.nn.L1Loss().to('cuda:0')
loss_Y = torch.nn.BCEWithLogitsLoss().to('cuda:0')

# TODO: add optimizer to LLM parameters
optimizer = torch.optim.Adam(list(ModelXtoCtoY_layer.parameters()), lr=1e-5)

best_acc_score = 0
for epoch in range(num_epochs):
    ######### train #########
    ModelXtoCtoY_layer.train()
    model.train()
    
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        concept_labels = batch['concept_labels']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to('cuda:0'), attention_mask=attention_mask.to('cuda:0'), output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:,0]

        outputs = ModelXtoCtoY_layer(pooled_output.to('cuda:0'))
        if experiment == 'baseline':
            XtoY_output = outputs.squeeze()
            XtoY_loss = loss_Y(XtoY_output, label.squeeze().to('cuda:0'))

            loss = XtoY_loss
        else:
            XtoC_output = outputs[1:] 
            XtoY_output = outputs[0:1]

            # XtoC_loss
            XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
            XtoC_loss = loss_C(XtoC_output, concept_labels.squeeze().to('cuda:0'))
        
            # XtoY_loss
            XtoY_loss = loss_Y(XtoY_output[0].squeeze(), label.squeeze().to('cuda:0'))
        
            loss = XtoY_loss + XtoC_loss * 0.5
        
        loss.backward()
        optimizer.step()

    ######### val #########
    model.eval()
    ModelXtoCtoY_layer.eval()
    val_accuracy = 0.
    concept_val_loss = 0.
    predict_labels = np.array([])

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.to('cuda:0'), attention_mask=attention_mask.to('cuda:0'), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)
            if experiment == 'baseline':
                XtoY_output = outputs

                predict_labels = np.append(predict_labels, (XtoY_output.squeeze().cpu() > 0.5) == label.bool().cpu())
            else:
                XtoC_output = outputs[1:] 
                XtoY_output = outputs[0:1]

                XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
                XtoC_loss = loss_C(XtoC_output, concept_labels.squeeze().to('cuda:0'))

                concept_val_loss += XtoC_loss.sum().item()
            
                predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().cpu() > 0.5) == label.bool().cpu())

        val_accuracy = predict_labels.sum() / len(predict_labels)
        
        if experiment != 'baseline':
            concept_val_loss = concept_val_loss / len(predict_labels)
        
    if experiment == 'baseline':
        print(f'Epoch {epoch + 1}: Val Acc = {val_accuracy*100}')
    else:
        print(f'Epoch {epoch + 1}: Val Acc = {val_accuracy*100}')
        print(f'Epoch {epoch + 1}: Val concept MAE = {concept_val_loss}')

    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, 'models/joint.pth')
        torch.save(ModelXtoCtoY_layer, 'models/ModelXtoCtoY_layer_joint.pth')

######### test #########
num_epochs = 1
model = torch.load('models/joint.pth', weights_only=False)
ModelXtoCtoY_layer = torch.load('models/ModelXtoCtoY_layer_joint.pth', weights_only=False) 

for epoch in range(num_epochs):
    predict_labels = np.array([])
    true_labels = np.array([])
    predictions = np.array([])
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.to('cuda:0').squeeze(), attention_mask=attention_mask.to('cuda:0').squeeze(), output_hidden_states=True)

            pooled_output = outputs.hidden_states[-1][:,0] 

            outputs = ModelXtoCtoY_layer(pooled_output)
            if experiment == 'baseline':
                XtoY_output = outputs
                predictions = np.append(predictions, XtoY_output.squeeze().cpu())
                predict_labels = np.append(predict_labels, (XtoY_output.squeeze().cpu() > 0.5) == label.bool().cpu())
            else:
                XtoY_output = outputs[0:1]
                predictions = np.append(predictions, XtoY_output[0].squeeze().cpu())
                predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().cpu() > 0.5) == label.bool().cpu())

            true_labels = np.append(true_labels, label.bool().cpu())

        test_accuracy = predict_labels.sum() / len(predict_labels)

    print(f'Test Acc = {test_accuracy*100}')
    print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')