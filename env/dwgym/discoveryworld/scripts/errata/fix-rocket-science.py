import json
from glob import glob


def extract_action(gamestep):
    action = gamestep['nextAction']['action']
    arg1 = gamestep['nextAction'].get('arg1_desc', "")
    arg2 = gamestep['nextAction'].get('arg2_desc', "")
    return (action, arg1, arg2)


def fix_final_score_not_rocket_science(trajectory):

    pendulum_activated = False
    use_launch_terminal = False
    enter_orbit_velocity = False
    measured_light_angles = []
    measure_light_angle_twice = False
    read_pendulum = False
    use_load_cell_interface = False

    for gamestep in trajectory:
        if "nextAction" in gamestep and "action" in gamestep["nextAction"]:

            action, arg1, arg2 = extract_action(gamestep)

            if action == "ACTIVATE" and "pendulum" in arg1:
                pendulum_activated = True

            if action == "READ" and "pendulum" in arg1:
                read_pendulum = True

            if action == "USE" and "computer" in arg1:
                use_launch_terminal = True

            if action == "USE" and "speed square" in arg1:
                assert False, "tell Marc about this as I would need to check if the two measurements happened at valid places."
                measure_light_angle_twice += 1

            if action == "USE" and "load cell" in arg1:
                assert False, "tell Marc about this as I would need to check if load was actually refuled."
                use_load_cell_interface = True

            # NOTE: enter correct orbit velocity should already be computed correctly.

        if "oracle_scorecard" in gamestep:
            max_score = 0
            score = 0
            for scorecard in gamestep["oracle_scorecard"][0]["scoreCard"]:
                if scorecard['name'] == "Visit control room":
                    scorecard['score'] = 0
                    scorecard['maxScore'] = 0       # Removed from the scorecard, since the agent starts in the control room

                if scorecard['name'] == "Use launch terminal":
                    scorecard['score'] = int(use_launch_terminal)

                if scorecard['name'] == "Enter correct orbit velocity":
                    scorecard['score'] = int(enter_orbit_velocity)

                if scorecard['name'] == "Measure light angle":
                    if len(measured_light_angles) == 1:
                        scorecard['score'] = 1
                    elif len(measured_light_angles) >= 2:
                        scorecard['score'] = 2

                if scorecard['name'] == "Activate pendulum":
                    scorecard['score'] = int(pendulum_activated)

                if scorecard['name'] == "Observe pendulum":
                    scorecard['score'] = int(read_pendulum)

                if scorecard['name'] == "Use load cell interface":
                    scorecard['score'] = int(use_load_cell_interface)

                score += scorecard['score']
                max_score += scorecard['maxScore']

            gamestep["oracle_scorecard"][0]["score"] = score
            gamestep["oracle_scorecard"][0]["maxScore"] = max_score


def extract_final_score(trajectory):
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
dataPath = "paper-results/from-marc/discoveryworld-results/output_allhistory.It's (not) Rocket Science!-Normal-*"
#dataPath = "paper-results/from-marc/discoveryworld-results/output_allhistory.It's (not) Rocket Science!-Normal-*"
# dataPath = "paper-results/from-marc/discoveryworld-results/output_allhistory.Lost in Translation-Normal-*"
#dataPath = "paper-results/output-may27-hypothesizer-discovery-challenge-full5seeds/output_allhistory.Lost in Translation-Challenge-*"
filenames = sorted(glob(dataPath))

total_normalized_score = 0
total_completed = 0
total_completed_successfully = 0

for filename in filenames:
    with open(filename) as f:
        trajectory = json.load(f)

    print(f"----{f}----")
    fix_final_score_not_rocket_science(trajectory)
    normalized_score, completed, completedSuccessfully = extract_final_score(trajectory)
    total_normalized_score += normalized_score
    total_completed += completed
    total_completed_successfully += completedSuccessfully


print(f"Total normalized score: {total_normalized_score/len(filenames)}")
print(f"Total completed: {total_completed}/{len(filenames)} = {total_completed/len(filenames)*100:.2f}%")
print(f"Total completed successfully: {total_completed_successfully}/{len(filenames)} = {total_completed_successfully/len(filenames)*100:.2f}%")
