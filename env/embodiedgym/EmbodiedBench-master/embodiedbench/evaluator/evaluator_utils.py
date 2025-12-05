import json
import os
import glob

def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

def load_saved_data(path):
    dirs = os.listdir(path)
    json_files = glob.glob(os.path.join(path, '*.json')) 
    instructions = []
    for json_file in json_files:
        with open(json_file, 'r+') as f:
            for line in f:  
                data = json.loads(line)
                instructions.append(data['instruction'])
                break
    return instructions

