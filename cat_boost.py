import catboost
import pandas as pd
from sklearn.metrics import roc_auc_score
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import sys
from backbone_template import MLP
from utils import set_seed, agent
import ast

set_seed(int(sys.argv[4]))

data_type = sys.argv[1]
num_epochs = int(sys.argv[2])
experiment = sys.argv[3]
seed = int(sys.argv[4])

train_df = pd.read_csv(f'data/train_{data_type}.csv')
test_df = pd.read_csv(f'data/test_{data_type}.csv')
val_df = pd.read_csv(f'data/val_{data_type}.csv')

# drop columns that have only one unique value - requirement for catboost
cols_to_drop = set(train_df.columns[train_df.nunique(dropna=False) == 1])
cols_to_drop = cols_to_drop.union(set(val_df.columns[val_df.nunique(dropna=False) == 1]))
cols_to_drop = cols_to_drop.union(set(test_df.columns[test_df.nunique(dropna=False) == 1]))

train_df = train_df.drop(columns=list(cols_to_drop))
val_df = val_df.drop(columns=list(cols_to_drop))
test_df = test_df.drop(columns=list(cols_to_drop))

# choose 30 features with gemini agent
'''
columns = train_df.columns.tolist()
features = agent(data_type, columns, 30)
features = ast.literal_eval(features)
'''
features = ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'fr_aniline', 'fr_nitro_arom', 'fr_para_hydroxylation', 'fr_phenol', 'qed', 'NumAromaticRings', 'fr_sulfonamd', 'BertzCT', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FractionCSP3', 'fr_thiophene', 'fr_amide', 'SlogP_VSA2', 'EState_VSA2', 'RingCount', 'fr_halogen', 'NHOHCount', 'NOCount', 'fr_ether', 'MinEStateIndex', 'fr_aryl_methyl', 'SMR_VSA7']

features = features + ['Drug', 'Y']
train_df = train_df[features]
val_df = val_df[features]
test_df = test_df[features]

tokenizer = APETokenizer()
tokenizer.load_vocabulary('apetokenizer/tokenizer.json')
model = AutoModelForSequenceClassification.from_pretrained('mikemayuare/SMILY-APE-BBBP').to('cuda:0')
model.eval()

# tokenize the train, val and test sets
encoding_train = []
for item in train_df['Drug']:
    encoding_train.append(tokenizer(
        item,
        max_length=32,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="np",
    ))

encoding_val = []
for item in val_df['Drug']:
    encoding_val.append(tokenizer(
        item,
        max_length=32,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="np",
    ))

encoding_test = []
for item in test_df['Drug']:
    encoding_test.append(tokenizer(
        item,
        max_length=32,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="np",
    ))

# get the embeddings for the train, val and test sets
input_ids_train = torch.tensor([item['input_ids'] for item in encoding_train]).to('cuda:0')
attention_mask_train = torch.tensor([item['attention_mask'] for item in encoding_train]).to('cuda:0')
outputs_train = model(input_ids=input_ids_train, attention_mask=attention_mask_train, output_hidden_states=True).hidden_states[-1][:,0].detach().cpu().numpy()

input_ids_val = torch.tensor([item['input_ids'] for item in encoding_val]).to('cuda:0')
attention_mask_val = torch.tensor([item['attention_mask'] for item in encoding_val]).to('cuda:0')
outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val, output_hidden_states=True).hidden_states[-1][:,0].detach().cpu().numpy()

input_ids_test = torch.tensor([item['input_ids'] for item in encoding_test]).to('cuda:0')
attention_mask_test = torch.tensor([item['attention_mask'] for item in encoding_test]).to('cuda:0')
outputs_test = model(input_ids=input_ids_test, attention_mask=attention_mask_test, output_hidden_states=True).hidden_states[-1][:,0].detach().cpu().numpy()

if experiment == 'baseline':
    classifier = catboost.CatBoostClassifier(iterations=100, depth=2, learning_rate=0.1, loss_function='Logloss', random_state=seed, verbose=False)
    classifier.fit(outputs_train, y=train_df['Y'], eval_set=(outputs_val, val_df['Y']), verbose=False)
    predictions = classifier.predict_proba(outputs_test)[:, 1]
    print(roc_auc_score(test_df['Y'], predictions))
else:
    # train the classifiers on 'num_concepts' random properties
    num_concepts = 30
    classifiers = []
    predictions = np.empty((len(train_df), num_concepts))

    for i, col in enumerate(random.sample(list(train_df.drop(columns=['Drug', 'Y']).columns), num_concepts)):
        classifier = catboost.CatBoostRegressor(iterations=100, depth=2, learning_rate=0.1, loss_function='MAE', random_state=seed, verbose=False)
        classifier.fit(outputs_train, y=train_df[col], eval_set=(outputs_val, val_df[col]), verbose=False)
        predictions[:, i] = classifier.predict(outputs_train)
        classifiers.append(classifier)

    # create the dataset
    class MyDataset(Dataset):
        def __init__(self, predictions, split):
            self.data = predictions
            if split == 'train':
                self.labels = train_df['Y']
            elif split == 'test':
                self.labels = test_df['Y']

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return_dict = {
                'data': self.data[index],
                'label': self.labels[index]
            }
            return return_dict

    train_loader = DataLoader(MyDataset(predictions, 'train'), batch_size=32, shuffle=True)

    # phase 2: train the MLP to predict the ground truth
    mlp = MLP(input_dim=num_concepts, expand_dim=0)
    optimizer = torch.optim.Adam(list(mlp.parameters()), lr=1e-5)
    loss_Y = torch.nn.BCEWithLogitsLoss()

    mlp.train()
    for i in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch['data'].float())
            loss = loss_Y(outputs.squeeze(), batch['label'])
            loss.backward()
            optimizer.step()
        
    test_predictions = np.empty((len(test_df), num_concepts))
    for i, classifier in enumerate(classifiers):
        test_predictions[:, i] = classifier.predict(outputs_test)

    test_predictions = np.array(test_predictions)

    test_loader = DataLoader(MyDataset(test_predictions, 'test'), batch_size=32, shuffle=False)

    ground_truth_predictions = np.array([])
    true_labels = np.array([])
    
    with torch.no_grad():
        mlp.eval()
        for batch in test_loader:
            output = mlp(batch['data'].float())
            true_labels = np.append(true_labels, batch['label'])
            ground_truth_predictions = np.append(ground_truth_predictions, output.numpy())

    print(f'Test roc_auc_score = {roc_auc_score(true_labels, ground_truth_predictions)}')