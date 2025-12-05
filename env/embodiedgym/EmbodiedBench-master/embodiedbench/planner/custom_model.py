import requests
import torch
import os
import io
import requests

temperature = 0
max_completion_tokens = 2048
server_url = os.environ.get('server_url')

class CustomModel():
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'
        

    def respond(self, prompt, obs=None):        
        with open(obs, "rb") as img_file:
            files = {"image": img_file}
            data = {"sentence": prompt}
            response = requests.post(server_url, files=files, data=data)

        res= response.json()['response']
        if response.status_code != 200:
            print("Error:", response.text)
        return res

