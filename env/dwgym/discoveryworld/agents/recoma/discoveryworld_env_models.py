from copy import deepcopy
import datetime
import json
import logging
import time

from attr import dataclass
from recoma.datasets.reader import DatasetReader, Example
from recoma.models.core.base_model import BaseModel
from recoma.models.core.generator import GenerationOutputs
from recoma.search.answerfromstate import AnswerFromState
from recoma.search.search import EarlyStoppingCondition
from recoma.search.state import SearchState
from pathlib import Path

from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from discoveryworld.ScenarioMaker import SCENARIO_INFOS, SCENARIO_NAMES

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryWorldExample(Example):
    scenario_name: str
    difficulty: str
    random_seed: int
    task_description: str

    @staticmethod
    def fields():
        return ["scenario_name", "difficulty", "random_seed", "task"]

    @property
    def unique_id(self):
        return self.scenario_name + "_" + self.difficulty + "_" + str(self.random_seed)

    @property
    def task(self):
        return self.task_description if self.task_description is not None else str(self)

    @property
    def label(self):
        return 1.0

    def __str__(self) -> str:
        return f"{self.scenario_name}-{self.difficulty}-{self.random_seed}: {self.task_description}"

class SingletonEnvironment:
    _shared_env: DiscoveryWorldAPI = None

    def __init__(self, env=None):
        if env != None:
            SingletonEnvironment._shared_env = env

    @property
    def env(self) -> DiscoveryWorldAPI:
        if SingletonEnvironment._shared_env is None:
            raise ValueError("Environment not set up!")
        return SingletonEnvironment._shared_env


@BaseModel.register("discoveryworld_loader")
class DiscoveryWorldLoaderModel(BaseModel):
    def __init__(self, threadid_offset=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threadid_offset = threadid_offset

    def generate_output(self, state: SearchState) -> GenerationOutputs:
        # Generate Thread ID based on using a 4 digit number: ABCD.
        # A is some number depending on easy/med/hard.  B is the seed.
        # CD is a number unique to the scenario.
        # A: 1 = easy, 2 = medium, 3 = challenge
        diff2ID = { "Easy": 1, "Normal": 2, "Challenge": 3, "Test": 4 }
        threadId = (diff2ID[state.example.difficulty] * 1000)
        # B: seed
        threadId += state.example.random_seed * 100
        # CD: scenario number
        assert len(SCENARIO_NAMES) < 100, "Change logic for threadID generation. >100 scenarios"
        threadId += SCENARIO_NAMES.index(state.example.scenario_name)
        # if an offset is passed, add it to the threadId at the 5th digit position
        threadId += self.threadid_offset * 10000
        singleton_env = SingletonEnvironment(env=DiscoveryWorldAPI(threadID=threadId))

        env = singleton_env.env
        success = env.loadScenario(scenarioName = state.example.scenario_name,
                                   difficultyStr = state.example.difficulty,
                                   randomSeed = state.example.random_seed,
                                    numUserAgents = 1)

        if not success:
            raise ValueError("Failed to load scenario: {}".format(state.example.unique_id))
        observation = env.getAgentObservation(agentIdx=0)
        task_description = "" #"Tasks:\n"
        for task in observation["ui"]["taskProgress"]:
            task_description += task["description"] + "\n"
        task_description = task_description.strip()
        state.example.task_description = task_description
        return GenerationOutputs(outputs=[task_description])


@BaseModel.register("discoveryworld_env")
class DiscoveryWorldEnvironmentModel(BaseModel):

    def __init__(self, output_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir

    def generate_output(self, state: SearchState) -> GenerationOutputs:
        current_node = state.get_open_node()
        if current_node is None:
            raise ValueError(
                "DiscoveryWorldEnvironmentModel called without any open node!!")
        env = SingletonEnvironment().env
        # execute the input against the DiscoveryWorld environment
        action_json = json.loads(current_node.input_str)
        output = env.performAgentAction(agentIdx=0, actionJSON=action_json)
        if self.output_dir is not None:
            output_tracking_info(output_dir=self.output_dir, state=state, action=action_json,
                                 result=output)
        # Perform world tick
        env.tick()

        return GenerationOutputs(outputs=[json.dumps(output)])


@AnswerFromState.register("discoveryworld_answerer")
class DiscoveryWorldAnswerer(AnswerFromState):

    def __init__(self, output_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir

    def generate_answer(self, state: SearchState):
        api = SingletonEnvironment().env
        finalScorecard = api.getTaskScorecard()
        print("Final scorecard: ")
        print(json.dumps(finalScorecard, indent=4, sort_keys=True))
        state.data["num_steps"] = api.getStepCounter()
        state.data["final_scorecard"] = finalScorecard
        if self.output_dir is not None:
            filenameOut = self.output_dir + "/{}.mp4".format(state.example.unique_id)
            api.createAgentVideo(agentIdx=0, filenameOut=filenameOut)
            # also append the final scorecard
            output_tracking_info(output_dir=self.output_dir, state=state, action=None, result=None)
            save_world_state(state, self.output_dir)
        # taskName = finalScorecard[0]["taskName"]
        finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
        # print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
        # print("Number of steps: " + str(api.getStepCounter()))
        return str(finalNormalizedScore)


@DatasetReader.register("discoveryworld_reader")
class DiscoveryWorldReader(DatasetReader):
    def __init__(self, limit_prefixes=None, limit_difficulties=None, limit_seeds=None, **kwargs):
        super().__init__(**kwargs)
        self.limit_prefixes = limit_prefixes
        self.limit_difficulties = limit_difficulties
        self.limit_seeds = limit_seeds

    def read_examples(self, file: str):
        for scenarioName in SCENARIO_NAMES:
            accept_scenario = False
            if self.limit_prefixes is not None:
                for prefix in self.limit_prefixes:
                    if scenarioName.startswith(prefix):
                        accept_scenario = True
                        break
            else:
                accept_scenario = True

            if not accept_scenario:
                continue
            # Get the valid difficulty settings and random seeds for this scenario
            validDifficulties = SCENARIO_INFOS[scenarioName]["difficulty"]
            validSeeds = SCENARIO_INFOS[scenarioName]["variations"]
            validSeeds = [int(x) for x in validSeeds]
            for difficulty in validDifficulties:
                if self.limit_difficulties is not None and difficulty not in self.limit_difficulties:
                    continue
                for seed in validSeeds:
                    if self.limit_seeds is not None and seed not in self.limit_seeds:
                        continue
                    yield DiscoveryWorldExample(scenario_name=scenarioName,
                                                difficulty=difficulty,
                                                random_seed=seed,
                                                task_description=None)


@EarlyStoppingCondition.register("max_env_calls")
class MaximumEnvironmentCalls(EarlyStoppingCondition):
    def __init__(self, max_env_calls=200, **kwargs):
        super().__init__(**kwargs)
        self.max_env_calls = max_env_calls

    def should_stop(self, current_state: SearchState, num_iters: int, heap: list[SearchState]):
        env_calls = num_interactions()
        if env_calls >= self.max_env_calls:
            logger.warning("Hit max env calls: {}".format(env_calls))
            return True
        return False

def task_completed():
    env = SingletonEnvironment().env
    return env.areTasksComplete()

def num_interactions():
    env = SingletonEnvironment().env
    return env.getStepCounter()

def output_tracking_info(output_dir, state, action, result):
    # also append the final scorecard
    filenameOut = output_dir + "/{}_tracking.jsonl".format(state.example.unique_id)
    api = SingletonEnvironment().env
    scorecard = api.getTaskScorecard()
    # Can't jsonify Success object
    if result and "success" in result:
        try:
            result = deepcopy(result)
        except Exception as e:
            print(result)
            print("Error copying result: " + str(e))
            raise e

        result["success"] = str(result["success"])
    output_json = {
     "step": num_interactions(),
     "observation": api.getAgentObservation(agentIdx=0),
     "scorecard": scorecard,
     "action": action,
     "result": result
    }
    mode = "a" if num_interactions() > 1 else "w"
    with open(filenameOut, mode) as output_fp:
        output_fp.write(json.dumps(output_json) + "\n")


def save_world_state(state, output_dir):
    api = SingletonEnvironment().env
    scenarioName = state.example.scenario_name
    difficultyStr = state.example.difficulty
    seed = state.example.random_seed
    # Save log file
    logFileSuffix = scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-thread" + str(api.THREAD_ID)
    # Add date and time stamp
    logFileSuffix += "." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    verboseLogDirectory = output_dir + "/logs/" + logFileSuffix
    Path(verboseLogDirectory).mkdir(parents=True, exist_ok=True)
    logInfo = {
        "scenarioName": scenarioName,
        "difficulty": difficultyStr,
        "seed": seed,
        "numSteps": num_interactions(),
        "threadId": api.THREAD_ID,
        "dateStarted": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Make a verbose filename for the log
        "verboseLogDirectory": verboseLogDirectory,
        "verboseLogFilename": verboseLogDirectory + "/" + logFileSuffix + ".json",
    }
    print("Saving world history...")
    try:
        api.world.exportWorldHistoryJSON(logInfo, logInfo["verboseLogFilename"], None, None, None)
    except Exception as e:
        print("Error saving world history: " + str(e))