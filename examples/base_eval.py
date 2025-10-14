import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import argparse
from controller import APIAgentvllm
from clients import SqlGymEnvClient

def main(args):

    apiagentvllm = APIAgentvllm(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model_name,
        temperature=args.temperature,
    )

    env_client = SqlGymEnvClient(env_server_base=args.env_server_base, data_len=200, timeout=args.timeout)

    obs, info = env_client.reset(0)
    system_prompt = env_client.conversation_start
    while True:
        prompt = system_prompt[0]["value"]
        prompt += "\n"
        prompt += obs

        response = apiagentvllm.generate(prompt)

        step_output = env_client.step(response)

        obs = step_output.state
        reward = step_output.reward
        done = step_output.done

        if done:
            break

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_name', type=str, default="Qwen3-30B-Instruct")
    parse.add_argument('--api_key', type=str, default="EMPTY")
    parse.add_argument('--base_url', type=str, default="http://localhost:8001/v1")
    parse.add_argument('--temperature', type=float, default=0.3)
    parse.add_argument('--timeout', type=int, default=300)
    parse.add_argument('--env_server_base', type=str, default="http://127.0.0.1:36001")
    args = parse.parse_args()
    print(args)
    
    main(args)