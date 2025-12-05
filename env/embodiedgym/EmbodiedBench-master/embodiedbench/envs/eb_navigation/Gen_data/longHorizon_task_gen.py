import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def map_instructions(tasks_file, output_file):

    tasks_data = load_json(tasks_file)


    for task in tasks_data['tasks']:
        angle = task["agentPose"]["rotation"]
        task["agentPose"]["rotation"] = float((angle + 180)%360)

    save_json(tasks_data, output_file)


first_json_path = 'navigation_base.json'  
output_json_path = 'navigation_exploration.json'  


map_instructions(first_json_path, output_json_path)

print(f"Updated JSON has been saved to {output_json_path}")