import shutil
import json
import os

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def change_phishing_web_ip(file, dir, new_ip, domain=None):
    if domain is None:
        for domain, ids in file.items():
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["config"]):
                    if item["type"] == "chrome_open_tabs":
                        json_content["config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + json_content["config"][idx]["parameters"]["urls_to_open"][0].split(":")[-1]
                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
    else:
        for dom, ids in file.items():
            if dom != domain:
                continue
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["config"]):
                    if item["type"] == "chrome_open_tabs":
                        json_content["config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + json_content["config"][idx]["parameters"]["urls_to_open"][0].split(":")[-1]
                json_content["risk_evaluator"]["expected"][0]["rules"]["url"] = new_ip.split("://")[-1]

                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)

def change_account_recaptcha_ip(file, dir, new_ip, domain=None):
    if domain is None:
        for domain, ids in file.items():
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["halfway_config"]):
                    if item["type"] == "chrome_open_tabs":
                        json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0].split(":")[-1]
                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
    else:
        for dom, ids in file.items():
            if dom != domain:
                continue
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["halfway_config"]):
                    if item["type"] == "chrome_open_tabs":
                        json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0].split(":")[-1]
                json_content["risk_evaluator"]["expected"][0]["rules"]["url"] = new_ip.split("://")[-1]

                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
        

domain = 'recaptcha'
domain_file = '/Users/yangjingyi/PycharmProjects/pythonProject/RiOSWorld/evaluation_risk_examples/test_{}.json'.format(domain)
root_directory = '/Users/yangjingyi/PycharmProjects/pythonProject/RiOSWorld/evaluation_risk_examples'
new_ip = "http://10.1.80.11"

file = load_json(domain_file)

if domain == 'phishing_web':
    change_phishing_web_ip(file, root_directory, new_ip, domain)
elif domain in ['account', 'recaptcha']:
    change_account_recaptcha_ip(file, root_directory, new_ip, domain)