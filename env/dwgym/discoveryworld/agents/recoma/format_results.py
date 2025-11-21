from ast import List
import json
from multiprocessing.managers import ListProxy
from pathlib import Path
import re
import sys

from attr import dataclass


@dataclass
class TaskResult:
    scenario: str
    difficulty: str
    seed: str
    max_iters: int
    completed: float
    completed_successfully: float
    normalized_score: float
    num_interactions: int
    cost: float

    @staticmethod
    def fields():
        return ["scenario", "difficulty", "seed", "max_iters", "normalized_score",
                "completed", "completed_successfully", "num_interactions", "cost"]

    @staticmethod
    def aggregate(results):
        count = normalized_score = completed = completed_successfully = num_interactions = cost = 0
        seeds = []
        for result in results:
            normalized_score += result.normalized_score
            completed += 1 if result.completed else 0
            completed_successfully += 1 if result.completed_successfully else 0
            num_interactions += result.num_interactions
            cost += result.cost
            seeds.append(result.seed)
            count += 1
        result = results[0]
        return TaskResult(result.scenario, result.difficulty, str(seeds), result.max_iters,
                            completed / count, completed_successfully / count,
                            normalized_score / count, num_interactions / count, cost / count)


    def to_tsv(self):
        formatted_fields = [f"{self.scenario}", f"{self.difficulty}", f"{self.seed}",
                             f"{self.max_iters}", f"{self.normalized_score:.2f}",
                             f"{self.completed}", f"{self.completed_successfully}",
                               f"{self.num_interactions}", f"{self.cost:.2f}"]
        return "\t".join(formatted_fields)


def extract_results(dir):
    m = re.match(r".*_(\d+)env.*", dir)
    if m:
        max_iters = int(m.group(1))
    else:
        max_iters = 200

    # iterate through all *data.json files in dir
    for file in Path(dir).glob("*data.json"):
        # (scenario, difficulty, seed, ext) = file.name.split("_")
        with open(file, "r") as f:
            data = json.load(f)
            scenario = data["scenario_name"]
            difficulty = data["difficulty"]
            seed = data["random_seed"]
            scorecard = data["metadata"]["final_scorecard"][0]
            completed = scorecard["completed"]
            completed_successfully = scorecard["completedSuccessfully"]
            normalized_score = scorecard["scoreNormalized"]
            num_interactions = data["metadata"]["num_steps"]
            cost = sum([x[1] for x in data["metadata"].items() if x[0].endswith(".cost")])
            yield TaskResult(scenario, difficulty, seed, max_iters, completed,
                             completed_successfully, normalized_score, num_interactions, cost)


def main():
    dirs = sys.argv[1:]
    task_results = []
    for dir in dirs:
        task_results.extend(extract_results(dir))

    task_results.sort(key=lambda x: (x.scenario, x.difficulty, x.seed))

    print("\t".join(TaskResult.fields()))
    last_scenario_diff = None
    aggregate_results = []
    for task_result in task_results:
        scenario_diff = task_result.scenario + "_" +  task_result.difficulty
        if scenario_diff != last_scenario_diff:
            if len(aggregate_results) > 1:
                print(TaskResult.aggregate(aggregate_results).to_tsv())
            print("\n")
            last_scenario_diff = scenario_diff
            aggregate_results = [task_result]
        else:
            aggregate_results.append(task_result)
        print(task_result.to_tsv())
    # final print
    if len(aggregate_results) > 1:
        print(TaskResult.aggregate(aggregate_results).to_tsv())


if __name__ == "__main__":
    main()
