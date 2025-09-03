import sys
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import torch
from backbone import ModelXtoCtoY_function
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from utils import set_seed, agent
import ast
import pickle as pkl
from utils import MyDataset
import random

set_seed(int(sys.argv[4]))

# this is the tokenizer that we use for encoding SMILES strings
'''
model_name = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model_kwargs = {}
device = 'cuda'
if device == "cuda":
    try:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs.update(dict(quantization_config=quant, device_map="auto"))
    except Exception:
        model_kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
'''
model_name = 'mikemayuare/SMILY-APE-BBBP'
tokenizer = APETokenizer()
tokenizer.load_vocabulary('apetokenizer/tokenizer.json')
model = AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')

# get the data type and number of epochs from the command line
data_type = sys.argv[1]
num_epochs = int(sys.argv[2])
experiment = sys.argv[3]

# load data
DATA = {}
DATA['train'] = pd.read_csv(f'data/train_{data_type}.csv')
DATA['val'] = pd.read_csv(f'data/val_{data_type}.csv')
DATA['test'] = pd.read_csv(f'data/test_{data_type}.csv')

# choose num_concepts features with llm agent
num_concepts = int(sys.argv[5])
'''
features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
print(features)
features = ast.literal_eval(features)
'''
# choose features based on data type
if data_type == 'dili':
    if num_concepts == 30:
        if sys.argv[6] == 'random':
            features = random.sample(DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), 30)
        else:
            features = ['MolLogP', 'TPSA', 'MolWt', 'NumRotatableBonds', 'FractionCSP3', 'fr_aniline', 'fr_nitro_arom', 'NumAromaticRings', 'MaxAbsPartialCharge', 'qed', 'fr_thiophene', 'fr_furan', 'HeavyAtomCount', 'NumHDonors', 'NumHAcceptors', 'fr_quatN', 'fr_sulfonamd', 'RingCount', 'LabuteASA', 'fr_para_hydroxylation', 'fr_phenol', 'BertzCT', 'fr_halogen', 'fr_aryl_methyl', 'SlogP_VSA10', 'EState_VSA2', 'NumHeteroatoms', 'FpDensityMorgan2', 'MinAbsPartialCharge', 'fr_Ar_N']
    elif num_concepts == 50:
        features = ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors', 'HeavyAtomCount', 'NumAromaticRings', 'LabuteASA', 'fr_aniline', 'fr_nitro_arom', 'fr_para_hydroxylation', 'fr_phenol', 'fr_thiophene', 'fr_furan', 'fr_quatN', 'fr_sulfonamd', 'fr_amide', 'fr_Ar_N', 'fr_aryl_methyl', 'fr_epoxide', 'fr_C_O_noCOO', 'fr_ether', 'fr_halogen', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Ar_OH', 'fr_sulfone', 'fr_bicyclic', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'MaxEStateIndex', 'MinEStateIndex', 'FractionCSP3', 'qed', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA2', 'SlogP_VSA4', 'SlogP_VSA5', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA2', 'PEOE_VSA6', 'PEOE_VSA7', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2', 'EState_VSA5', 'EState_VSA8']
    elif num_concepts == 10:
        # features = random.sample(DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), 10)
        features = ['MolLogP', 'MolWt', 'TPSA', 'fr_aniline', 'NumRotatableBonds', 'fr_nitro_arom', 'fr_thiophene', 'fr_phenol', 'MaxAbsPartialCharge', 'fr_aldehyde']
elif data_type == 'bbbp':
    if num_concepts == 30:
        if sys.argv[6] == 'random':
            features = random.sample(DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), 30)
        else:
            features = ['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'FractionCSP3', 'qed', 'HeavyAtomCount', 'MolMR', 'fr_quatN', 'fr_COO', 'NHOHCount', 'LabuteASA', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsPartialCharge', 'SlogP_VSA3', 'SMR_VSA5', 'PEOE_VSA7', 'EState_VSA2', 'VSA_EState9', 'FpDensityMorgan2', 'BertzCT', 'RingCount', 'NumAromaticRings', 'NOCount', 'fr_aniline', 'fr_amide', 'fr_ether']
    elif num_concepts == 50:
        features = ['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'MolMR', 'LabuteASA', 'FractionCSP3', 'HeavyAtomCount', 'RingCount', 'NOCount', 'NHOHCount', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'qed', 'ExactMolWt', 'NumAromaticRings', 'NumHeteroatoms', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA8', 'EState_VSA1', 'EState_VSA2', 'EState_VSA10', 'VSA_EState9', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA5', 'SMR_VSA7', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Kappa2', 'Ipc', 'fr_amide', 'fr_aniline', 'fr_Ar_N', 'fr_COO', 'fr_ether', 'fr_phenol', 'fr_quatN', 'fr_unbrch_alkane']
    elif num_concepts == 10:
        features = ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'HeavyAtomCount', 'NOCount', 'FractionCSP3', 'qed']
elif data_type == 'lipo':
    if num_concepts == 30:
        features = ['MolLogP', 'TPSA', 'MolMR', 'LabuteASA', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA10', 'SMR_VSA1', 'SMR_VSA5', 'SMR_VSA7', 'SMR_VSA10', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'EState_VSA1', 'EState_VSA2', 'EState_VSA10', 'NumHAcceptors', 'NumHDonors', 'NOCount', 'HeavyAtomCount', 'NumRotatableBonds', 'FractionCSP3', 'RingCount', 'fr_halogen', 'fr_amide', 'fr_benzene', 'fr_ether']

means = DATA['train'][features].mean().values
stds = DATA['train'][features].std().values

# create the dataloader
train_loader = DataLoader(MyDataset('train', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=True)
val_loader = DataLoader(MyDataset('val', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)
test_loader = DataLoader(MyDataset('test', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)

# num_concepts is the number of concepts, expand_dim is the dimension of the expanded layer (0 means no expansion)
if experiment == 'baseline':
    ModelXtoCtoY_layer = torch.nn.Sequential(
        torch.nn.Linear(768, num_concepts),
        torch.nn.Linear(num_concepts, 1)
    )
else:
    ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0).to('cuda')

loss_C = torch.nn.L1Loss()
loss_Y = torch.nn.BCEWithLogitsLoss()

# TODO: add optimizer to LLM parameters
if experiment == 'baseline':
    optimizer = torch.optim.Adam(list(ModelXtoCtoY_layer.parameters()), lr=1e-5)
else:
    optimizer = torch.optim.Adam(list(ModelXtoCtoY_layer.parameters()) + list(model.parameters()), lr=1e-5)

best_acc_score = 0
for epoch in range(num_epochs):
    ######### train #########
    ModelXtoCtoY_layer.train()
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        label = batch['label'].to('cuda')
        concept_labels = batch['concept_labels']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:,0]

        outputs = ModelXtoCtoY_layer(pooled_output)
        if experiment == 'baseline':
            XtoY_output = outputs.squeeze()
            XtoY_loss = loss_Y(XtoY_output, label.squeeze())

            loss = XtoY_loss
        else:
            XtoC_output = outputs[1:] 
            XtoY_output = outputs[0:1]

            # XtoC_loss
            XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
            XtoC_loss = loss_C(XtoC_output.to('cuda'), concept_labels.squeeze().to('cuda'))
        
            # XtoY_loss
            XtoY_loss = loss_Y(XtoY_output[0].squeeze().to('cuda'), label.squeeze().to('cuda'))
        
            loss = XtoY_loss + XtoC_loss * float(sys.argv[8])
        
        loss.backward()
        optimizer.step()

    ######### val #########
    ModelXtoCtoY_layer.eval()
    model.eval()
    val_accuracy = 0.
    concept_val_loss = 0.
    predict_labels = np.array([])

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            label = batch['label'].to('cuda')
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)
            if experiment == 'baseline':
                XtoY_output = outputs

                predict_labels = np.append(predict_labels, (XtoY_output.squeeze().cpu() > 0.5) == label.bool().cpu())
            else:
                XtoC_output = outputs[1:] 
                XtoY_output = outputs[0:1]

                XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
                XtoC_loss = loss_C(XtoC_output.to('cuda'), concept_labels.squeeze().to('cuda'))

                concept_val_loss += XtoC_loss.sum().item()
            
                predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().cpu() > 0.5) == label.bool().cpu())

        val_accuracy = predict_labels.sum() / len(predict_labels)
        
        if experiment != 'baseline':
            concept_val_loss = concept_val_loss / len(predict_labels)
    '''
    if experiment == 'baseline':
        print(f'Epoch {epoch + 1}: Val Acc = {val_accuracy*100}')
    else:
        print(f'Epoch {epoch + 1}: Val Acc = {val_accuracy*100}')
        print(f'Epoch {epoch + 1}: Val concept MAE = {concept_val_loss}')
    '''
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, f'models/joint_{sys.argv[7]}_{sys.argv[8]}.pth')
        torch.save(ModelXtoCtoY_layer, f'models/ModelXtoCtoY_layer_joint_{sys.argv[7]}_{sys.argv[8]}.pth')

######### test #########
num_epochs = 1
model = torch.load(f'models/joint_{sys.argv[7]}_{sys.argv[8]}.pth', weights_only=False)
ModelXtoCtoY_layer = torch.load(f'models/ModelXtoCtoY_layer_joint_{sys.argv[7]}_{sys.argv[8]}.pth', weights_only=False) 
model.eval()
ModelXtoCtoY_layer.eval()

for epoch in range(num_epochs):
    predict_labels = np.array([])
    true_labels = np.array([])
    predictions = np.array([])
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            label = batch['label'].to('cuda')
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)
            if experiment == 'baseline':
                XtoY_output = outputs
                predictions = np.append(predictions, XtoY_output.squeeze().to(torch.float32).cpu())
                predict_labels = np.append(predict_labels, (XtoY_output.squeeze().to(torch.float32).cpu() > 0.0) == label.bool().cpu())
            else:
                XtoY_output = outputs[0:1]
                predictions = np.append(predictions, XtoY_output[0].squeeze().to(torch.float32).cpu())
                predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().to(torch.float32).cpu() > 0.0) == label.bool().cpu())

            true_labels = np.append(true_labels, label.bool().cpu())

        test_accuracy = predict_labels.sum() / len(predict_labels)

    with open(f'models/val_data_{data_type}_vanilla.pkl_{sys.argv[7]}_{sys.argv[8]}', 'wb') as f:
        pkl.dump(val_loader, f)
    print(f'Test Acc = {test_accuracy*100}')
    print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')