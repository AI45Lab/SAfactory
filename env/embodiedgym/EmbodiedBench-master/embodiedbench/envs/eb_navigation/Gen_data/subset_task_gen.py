import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def map_instructions(tasks_file, floorplan_file, output_file):

    tasks_data = load_json(tasks_file)
    floorplan_data = load_json(floorplan_file)


    for task in tasks_data['tasks']:
        scene = task.get('scene')
        if scene in floorplan_data:
            task['instruction'] = floorplan_data[scene]

    save_json(tasks_data, output_file)


first_json_path = 'navigation_base.json' 
second_json_path = 'instruct-common3.json'  
output_json_path = 'navigation_common_sense.json' 


map_instructions(first_json_path, second_json_path, output_json_path)

print(f"Updated JSON has been saved to {output_json_path}")
