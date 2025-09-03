import os
import random
import numpy as np
import torch
import sys
from google import genai
from google.genai import types
import os
from torch.utils.data import Dataset
from openai import OpenAI
from pydantic import BaseModel

def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def agent(task, properties, num_concepts):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if task == 'dili':
        task = 'Predict the DILi (Drug-Induced Liver Injury) risk of a molecule.'
    elif task == 'bbbp':
        task = 'Predict the BBBP (Blood-Brain Barrier Penetration) risk of a molecule.'
    elif task == 'lipo':
        task = 'Predict the LIPO (Lipophilicity) of a molecule.'

    class agent_response(BaseModel):
        selected_properties: list[str]

    response = client.responses.create(
        model = 'gpt-5',
        instructions = 'You are a helpful chemistry assistant. You will be given a task and some properties to choose from. You will choose the properties most important for successfully completing/predicting the task.',
        input = f'Task: {task}\nProperties: {properties}. You should choose {num_concepts} properties and return them in a list formatted exactly as they are given to you. The list should be your ONLY output.',
    )
    print(response)
    
    return response.text

# create the dataset for the vanilla model
class MyDataset(Dataset):
    def __init__(self, split, features, means, stds, tokenizer, DATA):
        self.data = DATA[split]
        self.features = features
        self.means = means
        self.stds = stds
        self.stds[self.stds == 0] = 1
        self.tokenizer = tokenizer
        self.labels = self.data['Y']
        self.text = self.data["Drug"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tokenized_text = self.tokenizer(
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
            'concept_labels': torch.tensor((self.data.iloc[index][self.features].values.astype(float) - self.means) / self.stds),
            'features': self.features
        }

        return return_dict