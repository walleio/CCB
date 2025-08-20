from torch_geometric.loader import DataLoader
import pandas as pd
import torch
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils.smiles import from_smiles
from sklearn.metrics import roc_auc_score
import numpy as np
from backbone_template import MLP
import random
import sys
from utils import set_seed, agent

data_type = sys.argv[1]
num_epochs = int(sys.argv[2])
experiment = sys.argv[3]

set_seed(int(sys.argv[4]))

num_concepts = int(sys.argv[5])

# molecules are PyG objects, so we need to attach the y and concepts to the object
def attach_y_and_concepts(row):
    row['Drug'].y = torch.tensor(row['Y'], dtype=torch.float32)
    try:
        row['Drug'].concepts = torch.tensor(np.asarray(row[features].values, dtype=np.float32))
    except:
        print(row[features].values)
    return row['Drug']

# load data
train_data = pd.read_csv(f"data/train_{data_type}.csv")
test_data = pd.read_csv(f"data/test_{data_type}.csv")
val_data = pd.read_csv(f"data/val_{data_type}.csv")

# choose num_samples features with gemini agent
'''
features = agent(data_type, train_data.drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
features = ast.literal_eval(features)
'''
if data_type == 'dili':
    if num_concepts == 30:
        features = ['MolLogP', 'TPSA', 'MolWt', 'NumRotatableBonds', 'FractionCSP3', 'fr_aniline', 'fr_nitro_arom', 'NumAromaticRings', 'MaxAbsPartialCharge', 'qed', 'fr_thiophene', 'fr_furan', 'HeavyAtomCount', 'NumHDonors', 'NumHAcceptors', 'fr_quatN', 'fr_sulfonamd', 'RingCount', 'LabuteASA', 'fr_para_hydroxylation', 'fr_phenol', 'BertzCT', 'fr_halogen', 'fr_aryl_methyl', 'SlogP_VSA10', 'EState_VSA2', 'NumHeteroatoms', 'FpDensityMorgan2', 'MinAbsPartialCharge', 'fr_Ar_N']
    elif num_concepts == 50:
        features ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors', 'HeavyAtomCount', 'NumAromaticRings', 'LabuteASA', 'fr_aniline', 'fr_nitro_arom', 'fr_para_hydroxylation', 'fr_phenol', 'fr_thiophene', 'fr_furan', 'fr_quatN', 'fr_sulfonamd', 'fr_amide', 'fr_Ar_N', 'fr_aryl_methyl', 'fr_epoxide', 'fr_C_O_noCOO', 'fr_ether', 'fr_halogen', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Ar_OH', 'fr_sulfone', 'fr_bicyclic', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'MaxEStateIndex', 'MinEStateIndex', 'FractionCSP3', 'qed', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA2', 'SlogP_VSA4', 'SlogP_VSA5', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA2', 'PEOE_VSA6', 'PEOE_VSA7', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2', 'EState_VSA5', 'EState_VSA8']
    elif num_concepts == 10:
        features = ['MolLogP', 'MolWt', 'TPSA', 'fr_aniline', 'NumRotatableBonds', 'fr_nitro_arom', 'fr_thiophene', 'fr_phenol', 'MaxAbsPartialCharge', 'fr_aldehyde']
elif data_type == 'bbbp':
    if num_concepts == 30:
        features = ['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'FractionCSP3', 'qed', 'HeavyAtomCount', 'MolMR', 'fr_quatN', 'fr_COO', 'NHOHCount', 'LabuteASA', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsPartialCharge', 'SlogP_VSA3', 'SMR_VSA5', 'PEOE_VSA7', 'EState_VSA2', 'VSA_EState9', 'FpDensityMorgan2', 'BertzCT', 'RingCount', 'NumAromaticRings', 'NOCount', 'fr_aniline', 'fr_amide', 'fr_ether']
    elif num_concepts == 50:
        features = ['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'MolMR', 'LabuteASA', 'FractionCSP3', 'HeavyAtomCount', 'RingCount', 'NOCount', 'NHOHCount', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'qed', 'ExactMolWt', 'NumAromaticRings', 'NumHeteroatoms', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA5', 'SlogP_VSA8', 'EState_VSA1', 'EState_VSA2', 'EState_VSA10', 'VSA_EState9', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA5', 'SMR_VSA7', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Kappa2', 'Ipc', 'fr_amide', 'fr_aniline', 'fr_Ar_N', 'fr_COO', 'fr_ether', 'fr_phenol', 'fr_quatN', 'fr_unbrch_alkane']
    elif num_concepts == 10:
        features = ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'HeavyAtomCount', 'NOCount', 'FractionCSP3', 'qed']
elif data_type == 'lipo':
    if num_concepts == 30:
        features = ['MolLogP', 'TPSA', 'MolMR', 'LabuteASA', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA10', 'SMR_VSA1', 'SMR_VSA5', 'SMR_VSA7', 'SMR_VSA10', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'EState_VSA1', 'EState_VSA2', 'EState_VSA10', 'NumHAcceptors', 'NumHDonors', 'NOCount', 'HeavyAtomCount', 'NumRotatableBonds', 'FractionCSP3', 'RingCount', 'fr_halogen', 'fr_amide', 'fr_benzene', 'fr_ether']

# turn the SMILES strings into PyG objects
train_data['Drug'] = train_data['Drug'].apply(from_smiles)
test_data['Drug'] = test_data['Drug'].apply(from_smiles)
val_data['Drug'] = val_data['Drug'].apply(from_smiles)  

# attach the y and concepts to the PyG objects
train_data['Drug'] = train_data.apply(attach_y_and_concepts, axis=1)
test_data['Drug'] = test_data.apply(attach_y_and_concepts, axis=1)
val_data['Drug'] = val_data.apply(attach_y_and_concepts, axis=1)

# create the data loaders
train_loader = DataLoader(train_data['Drug'], batch_size=32, shuffle=True)
val_loader = DataLoader(val_data['Drug'], batch_size=32, shuffle=False)
test_loader = DataLoader(test_data['Drug'], batch_size=32, shuffle=False)

# define the model
class MolNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MolNet, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_channels, hidden_channels))
        
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_channels, hidden_channels))
                                 
        # 3 is the number of edge features from the 'from_smiles' function
        self.conv1 = GINEConv(nn1, edge_dim=3)
        self.conv2 = GINEConv(nn2, edge_dim=3)
        self.readout = global_add_pool
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = self.conv2(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = self.readout(x, batch)
        return self.mlp(x)

# initialize the model, optimizer, and loss functions
if experiment == 'baseline':
    num_concepts = 1

model = MolNet(in_channels=train_data['Drug'][1].x.shape[1], hidden_channels=256, out_channels=num_concepts)
mlp = MLP(input_dim=num_concepts, expand_dim=0)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlp.parameters()), lr=2e-4)
loss_C = torch.nn.L1Loss()
loss_Y = torch.nn.BCEWithLogitsLoss()

# train the model
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        if experiment == 'baseline':
            loss = loss_Y(output.squeeze(), data.y)
        else:
            loss_c = loss_C(output.reshape(-1), data.concepts)
            loss_y = loss_Y(mlp(output).squeeze(), data.y)
            loss = loss_c * 0.5 + loss_y
        loss.backward()
        optimizer.step()

# test the model
with torch.no_grad():
    model.eval()
    mlp.eval()
    predictions = np.array([])
    true_labels = np.array([])
    for data in test_loader:
        output = model(data)
        if experiment == 'baseline':
            predictions = np.append(predictions, output.numpy())
        else:
            mlp_output = mlp(output)
            predictions = np.append(predictions, mlp_output.numpy())
        true_labels = np.append(true_labels, data.y.numpy())

    print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')
