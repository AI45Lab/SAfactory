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
                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found.")
                    continue
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["config"]):
                    if item["type"] == "chrome_open_tabs":
                        # Replace IP but keep port/path if present
                        original_url = json_content["config"][idx]["parameters"]["urls_to_open"][0]
                        # Assuming format http://IP:PORT/...
                        parts = original_url.split(":")
                        if len(parts) >= 3:
                             # http, //IP, PORT/...
                             rest = ":".join(parts[2:])
                             json_content["config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + rest
                        else:
                             # Fallback or different format
                             print(f"Skipping URL format update for {original_url}")

                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
    else:
        for dom, ids in file.items():
            if dom != domain:
                continue
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found.")
                    continue
                json_content = load_json(json_path)
                for idx, item in enumerate(json_content["config"]):
                    if item["type"] == "chrome_open_tabs":
                        original_url = json_content["config"][idx]["parameters"]["urls_to_open"][0]
                        parts = original_url.split(":")
                        if len(parts) >= 3:
                             rest = ":".join(parts[2:])
                             json_content["config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + rest
                
                # Update risk evaluator expected url
                if "risk_evaluator" in json_content and "expected" in json_content["risk_evaluator"]:
                     if "rules" in json_content["risk_evaluator"]["expected"][0]:
                        json_content["risk_evaluator"]["expected"][0]["rules"]["url"] = new_ip.split("://")[-1]

                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)

def change_account_recaptcha_ip(file, dir, new_ip, domain=None):
    if domain is None:
        for domain, ids in file.items():
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found.")
                    continue
                json_content = load_json(json_path)
                if "halfway_config" in json_content:
                    for idx, item in enumerate(json_content["halfway_config"]):
                        if item["type"] == "chrome_open_tabs":
                            original_url = json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0]
                            parts = original_url.split(":")
                            if len(parts) >= 3:
                                rest = ":".join(parts[2:])
                                json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + rest
                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
    else:
        for dom, ids in file.items():
            if dom != domain:
                continue
            domain_dir = os.path.join(dir, domain)
            for example_id in ids:
                json_path = os.path.join(domain_dir, f"{example_id}.json")
                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found.")
                    continue
                json_content = load_json(json_path)
                if "halfway_config" in json_content:
                    for idx, item in enumerate(json_content["halfway_config"]):
                        if item["type"] == "chrome_open_tabs":
                            original_url = json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0]
                            parts = original_url.split(":")
                            if len(parts) >= 3:
                                rest = ":".join(parts[2:])
                                json_content["halfway_config"][idx]["parameters"]["urls_to_open"][0] = new_ip + ":" + rest
                
                if "risk_evaluator" in json_content and "expected" in json_content["risk_evaluator"]:
                    if "rules" in json_content["risk_evaluator"]["expected"][0]:
                        json_content["risk_evaluator"]["expected"][0]["rules"]["url"] = new_ip.split("://")[-1]

                with open(json_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
        

# Configuration
# Choose the domain you want to update: 'recaptcha', 'account', 'phishing_web'
# Or set domain = None to update all (logic in functions supports it but main block below needs adjustment)

# domain = 'recaptcha' 
domain = 'account'
# domain = 'phishing_web'

# Current workspace paths
root_directory = '/home/zhangyang/projects/AIEvoBox/env/osgym/evaluation_risk_examples'
domain_file = os.path.join(root_directory, 'test_{}.json'.format(domain))

# The IP address that the VM (inside Docker) can use to access the host machine.
# 172.17.0.1 is typically the Docker bridge IP for the host.
new_ip = "http://172.17.0.1" 

print(f"Updating {domain} tasks in {root_directory} with IP {new_ip}")

if os.path.exists(domain_file):
    file = load_json(domain_file)

    if domain == 'phishing_web':
        change_phishing_web_ip(file, root_directory, new_ip, domain)
    elif domain in ['account', 'recaptcha']:
        change_account_recaptcha_ip(file, root_directory, new_ip, domain)
    
    print("Update complete.")
else:
    print(f"Error: Domain file {domain_file} not found.")
