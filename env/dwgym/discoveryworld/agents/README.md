# DiscoveryWorld Baseline Agents

This README describes how to run the DiscoveryWorld baseline agents. 

The official benchmarking in the paper runs each *task_theme* + *difficulty* combination (24 total), for seeds `0, 1, 2, 3, 4`.  Easy and Unit Tests were run to a maximum of 100 steps, while Normal and Challenge difficulties were run to a maximum of 1000 steps.  Links to download the data from those runs is available in the `/data` folder:

https://github.com/allenai/discoveryworld/tree/main/data

**Helpful hints on developing your own agent:**
- The Random Agent provides a good and straightforward code example for what the action space looks like, and how to implement the arguments for any special-case actions (like dialog, teleporting, or move actions using north/east/south/west arguments).
- The API provides helper functions (e.g. `getActionDescriptions()`) that are designed to show the JSON response format of the action space to LLM agents that can read JSON-formatted instructions (like GPT4).
- The Hypothesizer agent is an example of a long-prompt agent, that can be examined as an example of one possible way to provide observations and context to LLM, and seek a JSON-formatted response.
- When developing, it's recommended to develop on the Unit tests, and keep runs short and inexpensive before you're sure they're functional and error-free. 

# 1. Task Themes, Difficulties, and Seeds

## 1.1 Task Themes
The following task themes are recognized:

### Discovery Tasks
```
Proteomics
Combinatorial Chemistry
Archaeology Dating
Plant Nutrients
Reactor Lab
Lost in Translation
Space Sick
It's (not) Rocket Science!
```

### Unit Tests
```
Small Skills: Dialog Test
Small Skills: Pick and Place Test
Small Skills: Pick and Give Test
Small Skills: Instrument Measurement Test
Small Skills: Doors Test
Small Skills: Doors with Keys Test
Small Skills: Navigation in a House Test
Small Skills: Search Test
Small Skills: Discovery Feed Test
Small Skills: Moving Agents Test
```

### Tutorial
(The same tutorial screening task available in the human graphical user interface)
```
Tutorial
```

## 1.2 Difficulties
The following difficulties are supported: `Easy`, `Normal`, `Challenge`.

## 1.3 Seeds
The official benchmark supports the following 5 seeds: `0, 1, 2, 3, 4`. 

Placing seeds higher than this should generate additional parametric scenarios.  Seeds beyond 0-4 haven't been manually vetted, and (for some task themes and difficulties) may throw an assertion or other error if their parametric generation generates data that isn't solvable.  Until we add catches for this, if you plan on using a large number of seeds out of the official range, we'd recommend either (a) manually vetting the seed range you plan to use, or (b) wrapping the world generation in a try/catch, and skipping over any problematic seeds.

# 2. Agent Models

## 2.1 Random Agent (Code Example Agent)

The *Random Agent* is a random baseline that, at each step, picks a random action to take.  It's both: 
- A good check to make sure that tasks are sufficiently difficult to solve
- A readable agent that helps introduce the action space to those developing new agents. 

The random action selection code ( https://github.com/allenai/discoveryworld/blob/main/agents/RandomAgent.py#L150 ) is designed to be easily read, and help you get up-and-running creating new agents quickly.  Most actions in DiscoveryWorld take arguments, and most of the time, those arguments reference specific objects in the environment (like `use *thermometer* on *flower*`).  Random actions of this sort are generated here: https://github.com/allenai/discoveryworld/blob/main/agents/RandomAgent.py#L185

But, some actions in DiscoveryWorld take other arguments.  For example:
- Some move actions take a direction (`north`, `east`, `south`, or `west`) as an argument.
- A teleport-to-location action takes a specific name (e.g. `science lab`) as an argument.
- Accessing DiscoveryWorld posts may take numeric arguments representing Post IDs.

The code here helps illustrate how to use these actions: https://github.com/allenai/discoveryworld/blob/main/agents/RandomAgent.py#L199

Lastly, being in **dialog** is a special case action, because agent dialog is pre-scripted in the form of **dialog trees**, and the agent can only say one of the pre-selected options.  Code to check for (and handle) being in a dialog state is shown here: 
https://github.com/allenai/discoveryworld/blob/main/agents/RandomAgent.py#L114

Note that dialog actions are common in DiscoveryWorld, as working with many objects that look like computer terminals (from the *soil computers* to the *crystal reactors*) are also implemented with dialog trees -- so dialog actions are an important case to implement in your agent. 

### 2.1.1 Running
The *Random Agent* can be run from the root directory of the repository using the following command template:
```
python agents/RandomAgent.py --scenario=$TASK_THEME --difficulty=$DIFFICULTY --seed=$SEED --numSteps=$NUM_STEPS
```

For example (on the Space Sick Scenario): 
```
python agents/RandomAgent.py --scenario="Space Sick" --difficulty="Challenge" --seed=0 --numSteps=10
```

### 2.1.2 Output

The Random Agent will output a filename in the root directory with a verbose filename (e.g. `output_random_agent*.json`) with summary statistics of the run: 
```
{
    "agentName": "RandomAgent",
    "completed": 0,
    "completedSuccessfully": 0,
    "difficulty": "Challenge",
    "finalNormalizedScore": 0.0,
    "numSteps": 10,
    "scenario": "Space Sick",
    "seed": 0,
    "stepsPerSecond": 3.6337145107164717,
    "threadId": 6638643
}
```

Notes on scoring terminology: 
- `completed` vs `completedSuccessfully`: Some tasks are forced choice (e.g. the archaeology tasks, where the agent must plant a flag in a specific location to finish the task).  `completed` refers to whether that final choice was taken/whether the task has finished, regardless of whether the task was completed successfully or not.  `completedSuccessfully` specifically refers to whether the correct answer was chosen/whether the task was completed successfully, and should be taken as the **task completion score**.
- `finalNormalizedScore`: This is the **procedural progress score**. 


## 2.2 Hypothesizer Agent

### 2.2.1 Running
The *Hypothesizer Agent* can be run from the root directory of the repository using the following command template:
```
python discoveryworld/DiscoveryWorldAPIUsageExample.py --scenario=$TASK_THEME --difficulty=$DIFFICULTY --numSteps=$NUM_STEPS --seed=$SEED --model=$MODEL --video --maxCostDollars=$MAX_COST_DOLLARS
```

Here's a toy example of running:
- the `Combinatorial Chemistry` theme
- on `Normal` difficulty
- Parametric Seed `0`
- up to `10` steps
- using the `gpt-4o-2024-05-13` model
- with a maximum cost of $`5`.00:
```
python agents/HypothesizerAgent.py --scenario="Combinatorial Chemistry" --difficulty="Normal" --numSteps=10 --seed=0 --model=gpt-4o-2024-05-13 --video --maxCostDollars=5
```

Here's also an example on the Dialog unit test that often completes in under 20 steps:
```
python agents/HypothesizerAgent.py --scenario="Small Skills: Dialog Test" --difficulty="Normal" --numSteps=25 --seed=0 --model=gpt-4o-2024-05-13 --video --maxCostDollars=5
````

Note that the HypothesizerAgent looks for an OpenAI key in a text file in the root directory of the repository called `openai_key.txt`.



### 2.2.2 Output

On completion, you should see a brief note with summary statistics of the run (i.e. the number of tokens sent/received, etc). 

In the root directory of the repository, the following logs (with verbose filenames identifying the task theme, difficulty, and seed) should be produced:
- `output_allhistory*.json`: Details of the agent's observations, actions, knowledge, and scores at each time step
- `output_consolodatedKnowledge*.json`: Details of any knowledge consolodation steps (occurs every 10 steps, by default)
- `output_costAnalysis*.json`: Details of the cost of running the agent
- `logs/out*partXofY.zip`: Detailed world state logs of the environment at each timestep, archived automatically to save room, and internally split into archives of 100 steps each. Also includes frame grabs at each step.
- `output*.mp4`: **VIDEO** If you have `ffmpeg` installed, and the `--video` option is selected, the agent will automatically try to convert the framegrabs into a `mp4` video that can be easily viewed.

### 2.2.3 Components of interest

The following components may be of interest in the code:

#### Act-then-reflect Cycle: 

The act-then-reflect cycle is implemented in the `GPT4HypothesizerOneStep()` function. 

The prompt for the action step begins here: 
https://github.com/allenai/discoveryworld/blob/main/agents/HypothesizerAgent.py#L348

The prompt for the knowledge reflection step (that takes place after each environment step) is here:
https://github.com/allenai/discoveryworld/blob/main/agents/HypothesizerAgent.py#L555

The addition to the prompt for handling being in dialog: 
https://github.com/allenai/discoveryworld/blob/main/agents/HypothesizerAgent.py#L405

The knowledge consolidation step, that attempts to summarize/compress the agent's working memory every 10 environment steps, to prevent it from growing large: 
https://github.com/allenai/discoveryworld/blob/main/agents/HypothesizerAgent.py#L642

Boilerplate code for initializing a DiscoveryWorld environment, and running your agent in it: 
https://github.com/allenai/discoveryworld/blob/main/agents/HypothesizerAgent.py#L975

## 2.3 ReAct Agent and Plan+Execute Agent
The `ReAct` and `Plan+Execute` Agents are based on existing codebases, with code and instructions available in the `/recoma` subdirectory of this repository: 
https://github.com/allenai/discoveryworld/tree/main/agents/recoma/README.md
