# React and Plan&Execute Baselines

## Setup

1. Install the Recoma library

```shell
pip install git+https://github.com/allenai/recoma.git@b262ab5476841551c5d6874e8ddecef78c71a072
```

2. Set the OPENAI_API_KEY env variable

## React

To run React on the unit tests using GPT4-Turbo, run the following command from the root directory of the Github repo:

```shell
export DIFF=Easy         # Difficulty level
export TASK="Space Sick"   # Task
export MAX_ENV_CALLS=100   # Max env calls
export SEED=123            # GPT Seed
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
python agents/recoma/run_recoma.py \
   --output_dir ${OUTPUT_DIR} \
   --config agents/recoma/configs/react.jsonnet
```

The `SEED` is used for GPT only and `OUTPUT_DIR` is where the results and videos are saved. To see all the scorecards, see *scorecard.jsonl files. To see the final scorecards for all examples, see all_data.jsonl.

## Plan & Execute
To run Plan & Execute, use the plan_and_execute.jsonnet. For example to run the Space Sick task on the Normal difficulty against all random seeds
and upto 100 env calls using GPT4-o.

```shell
export DIFF=Easy         # Difficulty level
export TASK="Space Sick"   # Task
export MAX_ENV_CALLS=100   # Max env calls
export SEED=123            # GPT Seed
export OUTPUT_DIR=output_dir/pne/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
export MODEL=gpt-4o-2024-05-13
python agents/recoma/run_recoma.py \
  --output_dir ${OUTPUT_DIR} \
  --config agents/recoma/configs/plan_and_execute.jsonnet
```


## How to configure experiments

1. Run experiments on other scenarios by changing the env vars or directly modifying the `limit_prefixes`, `limit_difficulties` and `limit_seeds` in the config files. By default, experiments are run on all the scenario seeds.

2. Change the maximum react history by setting `max_history` in react.jsonnet. By default (when set to -1), it will use maximum possible history till it hits the max character limit (`max_output_length=10000` in the config file).

3. Change `max_env_calls` to run for more environment steps. If you think this will increase the cost, make sure to change `max_llm_cost` too.

4. If you are using TogetherAI or Azure, change `MODEL` and add appropriate keys as per LiteLLM documentation. E.g. for Azure, read https://docs.litellm.ai/docs/providers/azure

5. To change the prompt, see prompts in `prompt/` folder

## Output format
The output from each scenario is saved in its own file, e.g. the output for ReAct agent on Space Sick (seed=1) on Normal difficulty would be saved
in `output_dir/react/Normal_1000env_gpt-4o-2024-05-13_s123/Space_Sick/Space\ Sick_Normal_1_data.json`. The file contains information about the
scenario and key data such as the cost of LLM usage, number of steps and scorecards.


## All Experiments

**Make sure to install the ReComA library first**

### React
Following commands can be used to reproduce results for the ReAct agent.

#### Small Skills
```shell
export DIFF=Normal
export TASK="Small Skills"
export MAX_ENV_CALLS=100
export SEED=123 # Used for GPT
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react.jsonnet
```

#### Easy
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react.jsonnet
```


#### Normal
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Challenge
 export TASK=${t}
 export MAX_ENV_CALLS=1000
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react.jsonnet
```


#### Challenge
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Challenge
 export TASK=${t}
 export MAX_ENV_CALLS=1000
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/react.jsonnet
```



### Plan & Execute
Following commands can be used to reproduce results for the Plan & Execute agent. 

#### Small Skills
```shell
export DIFF=Normal
export TASK="Small Skills"
export MAX_ENV_CALLS=100
export SEED=123 # Used for GPT
export MODEL=gpt-4o-2024-05-13
export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/plan_and_execute.jsonnet
```

#### Easy
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Easy
 export TASK=${t}
 export MAX_ENV_CALLS=100
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/plan_and_execute.jsonnet
```


#### Normal
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Challenge
 export TASK=${t}
 export MAX_ENV_CALLS=1000
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/plan_and_execute.jsonnet
```


#### Challenge
```shell
for t in "Archaeology Dating" "Plant Nutrients" "Space Sick" "Combinatorial Chemistry" "Reactor Lab" "Lost in Translation" "Proteomics" "It's (not)"; do
 export DIFF=Challenge
 export TASK=${t}
 export MAX_ENV_CALLS=1000
 export SEED=123 # Used for GPT
 export MODEL=gpt-4o-2024-05-13
 export OUTPUT_DIR=output_dir/react/${DIFF}_${MAX_ENV_CALLS}env_${MODEL}_s${SEED}/${TASK// /_}
 python agents/recoma/run_recoma.py \
    --output_dir ${OUTPUT_DIR} \
    --config agents/recoma/configs/plan_and_execute.jsonnet
```


