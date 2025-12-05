import os
import json
import glob
import argparse

def average_json_values(json_dir, target_file='*.json', output_file='summary_all.json', selected_key=None):
    values_sum = {}
    counts = {}

    json_files = glob.glob(os.path.join(json_dir, target_file)) + glob.glob(os.path.join(json_dir, '*', target_file)) + glob.glob(os.path.join(json_dir, '*', '*', target_file))
    print(json_files, len(json_files))
    for json_file in json_files:
        print(json_file.split('running/')[1])
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(data[selected_key] if selected_key!= None else data)
            for key, value in data.items():
                if selected_key != None and key != selected_key:
                    continue
                if type(value) == str:
                    continue
                    
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                if key not in values_sum:
                    values_sum[key] = 0.0
                    counts[key] = 0
                values_sum[key] += value
                counts[key] += 1
    
    averages = {key: values_sum[key] / counts[key] for key in values_sum}
    print('final results: ' )
    print(averages)
    with open(os.path.join(json_dir, output_file), 'w') as f:
        json.dump(averages, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files to compute average values.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing JSON files')
    parser.add_argument('--target_file', default='*.json', type=str, help='target file name')
    parser.add_argument('--output_file', default='summary_all.json', type=str, help='output file name')
    args = parser.parse_args()

    average_json_values(args.directory, args.target_file, args.output_file)
