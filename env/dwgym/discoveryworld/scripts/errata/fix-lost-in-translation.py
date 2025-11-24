import json
from glob import glob

def fix_final_score_lost_in_translation(trajectory):
    # TODO: could make a copy of trajectory instead of in-place modification
    trajectoryIdx = -1
    if ("oracle_scorecard" not in trajectory[trajectoryIdx]):
        # Look for it in the previous steps
        trajectoryIdx = -2
    if ("oracle_scorecard" not in trajectory[trajectoryIdx]):
        print("ERROR: The scorecard is not present in the last two steps.")
        exit(1)
    #assert len(trajectory[-1]["oracle_scorecard"]) == 1

    for idx, gamestep in enumerate(trajectory):
        if "oracle_scorecard" not in gamestep:
            continue

        display = False
        if (idx == len(trajectory)+trajectoryIdx):
            display = True

        max_score = 0
        score = 0
        for scorecard in gamestep["oracle_scorecard"][0]["scoreCard"]:
            if scorecard['name'] in ("Collect objects", "Give back objects"):
                # print score before
                if (display):
                    print(f"  BEFORE: {scorecard['name']}: {scorecard['score']}/{scorecard['maxScore']}")
                if scorecard['score'] >= scorecard['maxScore']:
                    scorecard['score'] = 2
                elif scorecard['score'] > 0:
                    scorecard['score'] = 1

                scorecard['maxScore'] = 2
                # print score after
                if (display):
                    print(f"   AFTER: {scorecard['name']}: {scorecard['score']}/{scorecard['maxScore']}")

            else:
                # print score anyway
                if (display):
                    print(f"NOCHANGE: {scorecard['name']}: {scorecard['score']}/{scorecard['maxScore']}")

            score += scorecard['score']
            max_score += scorecard['maxScore']

        gamestep["oracle_scorecard"][0]["score"] = score
        gamestep["oracle_scorecard"][0]["maxScore"] = max_score


def extract_final_score_lost_in_translation(trajectory):
    trajectoryIdx = -1
    if ("oracle_scorecard" not in trajectory[trajectoryIdx]):
        # Look for it in the previous steps
        trajectoryIdx = -2
    if ("oracle_scorecard" not in trajectory[trajectoryIdx]):
        print("ERROR: The scorecard is not present in the last two steps.")
        exit(1)

    #assert len(trajectory[-1]["oracle_scorecard"]) == 1
    print(f"#steps: {len(trajectory)}")
    print(f"#lastStepId: {trajectory[trajectoryIdx]['observation']['ui']['world_steps']}")
    assert len(trajectory[trajectoryIdx]["oracle_scorecard"]) == 1
    for scorecard in trajectory[trajectoryIdx]["oracle_scorecard"][0]["scoreCard"]:
        print(f"{scorecard['name']}: {scorecard['score']}/{scorecard['maxScore']}")

    normalized_score = trajectory[trajectoryIdx]["oracle_scorecard"][0]["score"] / trajectory[trajectoryIdx]["oracle_scorecard"][0]["maxScore"]
    print("Normalized score: " + str(normalized_score))

    # Also extract whether the task was completed (f)
    completed = 0
    completedSuccessfully = 0
    if (trajectory[trajectoryIdx]["oracle_scorecard"][0]["completed"] == True):
        completed = 1
    if (trajectory[trajectoryIdx]["oracle_scorecard"][0]["completedSuccessfully"] == True):
        completedSuccessfully = 1

    print("Completed: " + str(completed))
    print("Completed successfully: " + str(completedSuccessfully))

    return normalized_score, completed, completedSuccessfully


# Test the functions
# Test the functions
#dataPath = "./experiments/output_allhistory.It's (not) Rocket Science!-Normal-*"
#dataPath = "paper-results/from-marc/discoveryworld-results/output_allhistory.Lost in Translation-Normal-*"
dataPath = "paper-results/output-may27-hypothesizer-discovery-challenge-full5seeds/output_allhistory.Lost in Translation-Challenge-*"
filenames = sorted(glob(dataPath))


total_normalized_score = 0
total_completed = 0
total_completed_successfully = 0

for filename in filenames:
    with open(filename) as f:
        trajectory = json.load(f)

    print(f"----{f}----")
    fix_final_score_lost_in_translation(trajectory)
    normalized_score, completed, completedSuccessfully = extract_final_score_lost_in_translation(trajectory)
    total_normalized_score += normalized_score
    total_completed += completed
    total_completed_successfully += completedSuccessfully


print(f"Total normalized score: {total_normalized_score/len(filenames)}")
print(f"Total completed: {total_completed}/{len(filenames)} = {total_completed/len(filenames)*100:.2f}%")
print(f"Total completed successfully: {total_completed_successfully}/{len(filenames)} = {total_completed_successfully/len(filenames)*100:.2f}%")
