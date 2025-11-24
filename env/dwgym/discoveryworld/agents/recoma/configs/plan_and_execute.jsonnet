local model = std.extVar('MODEL');
local output_dir = std.extVar('OUTPUT_DIR');
local max_env_calls = std.parseInt(std.extVar('MAX_ENV_CALLS'));
local task = std.extVar('TASK');
local generator_params = import "default_greedy_generator.libsonnet";
{
    "models": {
        "discoveryworld_init": {
            "type": "discoveryworld_loader",
            "threadid_offset": 200 + (max_env_calls / 100), #200 used to indicate p&e
            "next_model": "discworld_plan_exec"
        },
        "discworld_plan_exec": {
            "type": "discoveryworld_plan_and_execute",
            "plan_model": "discworld_planner",
            "exec_model": "discworld_exec",
        },
        "discworld_planner": {
            "type": "discoveryworld_promptedlm",
            "prompt_file": "agents/recoma/prompts/planner_prompt.txt",
            "generator_params": generator_params + {"max_tokens": 100, "stop": ["\n"]},
        },
        "discworld_exec": {
            "type": "discoveryworld_react_controller",
            "action_model": "action",
            "observation_model": "environment",
            "add_roles": true,
            "max_output_length": 10000,
            "termination_suggestion": max_env_calls / 5,
        },
        "action": {
            "type": "discoveryworld_promptedlm",
            "prompt_file": "agents/recoma/prompts/executer_prompt.txt",
            "generator_params": generator_params + {"max_tokens": 400, "stop": ["```\n"]},
        },
        "environment": {
            "type": "discoveryworld_env",
            "output_dir": output_dir,
        },
    },
    "search": {
        "type": "best_first",
        "start_model": "discoveryworld_init",
        "answerer": {
            "type": "discoveryworld_answerer",
            "output_dir": output_dir,
        },
        "stopping_conditions": [
            {"type": "max_env_calls", "max_env_calls": max_env_calls},
            {"type": "max_llm_calls", "max_llm_calls": 1000}, // Not necessary; mainly there to catch any rogue usage
            {"type": "max_llm_cost", "max_llm_cost": 50.00}  // Not necessary; mainly there to catch any rogue usage
        ]
    },
    "reader": {
       "type": "discoveryworld_reader",
       "limit_prefixes": [task],
       "limit_difficulties": [std.extVar("DIFF")],
       "limit_seeds": [1, 2, 3, 4, 5],
    }
}