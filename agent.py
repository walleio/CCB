from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

def agent(task, properties, num_concepts):
    if task == 'dili':
        task = 'Predict the DILi (Drug-Induced Liver Injury) risk of a molecule.'
    elif task == 'bbbp':
        task = 'Predict the BBBP (Blood-Brain Barrier Penetration) risk of a molecule.'
    elif task == 'lipo':
        task = 'Predict the LIPO (Lipophilicity) of a molecule.'

    response = client.models.generate_content(
        model='gemini-2.5-pro',
        config=types.GenerateContentConfig(
            system_instruction='You are a helpful chemistry assistant. You will be given a task and some properties to choose from. You will choose the properties most important for successfully completing/predicting the task.'),
        contents=f'Task: {task}\nProperties: {properties}. You should choose {num_concepts} properties and return them in a list formatted exactly as they are given to you. The list should be your ONLY output.'
    )

    return response.text