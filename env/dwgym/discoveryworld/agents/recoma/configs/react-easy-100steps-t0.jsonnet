local model = std.extVar('MODEL');
local output_dir = std.extVar('OUTPUT_DIR');
//local max_env_calls = std.parseInt(std.extVar('MAX_ENV_CALLS'));
local max_env_calls = 100;
local generator_params = import "default_greedy_generator.libsonnet";
{
    "models": {
        "discoveryworld_init": {
            "type": "discoveryworld_loader",
            "threadid_offset": max_env_calls / 100,
            "next_model": "react"
        },
        "react": {
            "type": "discoveryworld_react_controller",
            "action_model": "action",
            "observation_model": "environment",
            "add_roles": true,
            "max_output_length": 10000,
            "max_history": -1  // Maximum number of history steps to show
        },
        "action": {
            "type": "discoveryworld_promptedlm",
            "prompt_file": "agents/recoma/prompts/react_prompt.txt",
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
        // "limit_prefixes": ["Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab", "Lost in Translation", "Space Sick", "Proteomics"],
        //"limit_prefixes": ["Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab"],
        // "limit_prefixes": ["Archaeology Dating"],        
        // "limit_prefixes": ["Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab", "Lost in Translation", "Space Sick", "Proteomics", "It's (not) Rocket Science!"],
        //  "limit_prefixes": ["Small Skills"],
        "limit_prefixes": ["Combinatorial Chemistry"],
        //"limit_prefixes": ["Archaeology Dating"],
        //"limit_prefixes": ["Plant Nutrients"],
        //"limit_prefixes": ["Reactor Lab"],
        //"limit_prefixes": ["Lost in Translation"],
        //"limit_prefixes": ["Space Sick"],
        //"limit_prefixes": ["Proteomics"],
        //"limit_prefixes": ["It's (not) Rocket Science!"],
        "limit_difficulties": ["Easy"],
        "limit_seeds": [1, 2, 3, 4, 5],
    }
}
