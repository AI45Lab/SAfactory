from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from discoveryworld.ScenarioMaker import SCENARIOS, SCENARIO_DIFFICULTY_OPTIONS

import json

# Given an observation, return an action to take
def model(api, observation, lastActionResult):
    prompt = "You are an agent, and must choose an action to perform a task towards a goal."

    # Add the task description to the prompt
    prompt += "Your task is: \n"
    taskDescription = observation["ui"]["taskProgress"][0]["description"]
    prompt += taskDescription + "\n"

    # Here is what you currently see:
    prompt += "You see: \n"
    prompt += "```json\n"
    prompt += json.dumps(observation["ui"], indent=4, sort_keys=True)
    prompt += "```\n"

    # Here are the actions you can take:
    prompt += "You can take the following actions: \n"
    prompt += "```json\n"
    prompt += json.dumps(api.listKnownActions(limited=False), indent=4, sort_keys=True)
    prompt += "```\n"

    # Teleport: Provide a list of valid teleport locations for this task
    prompt += "Teleporting: To make moving easier, you can teleport to a list of specific locations in the environment, using the teleport action.  In this case, 'arg1' is the name of a location, from the list below. An example teleport action would be: `{\"action\": \"TELEPORT_TO_LOCATION\", \"arg1\": \"school\"}).\n"
    prompt += "```json\n"
    prompt += json.dumps(api.listTeleportLocationsDict(), indent=4, sort_keys=True)
    prompt += "```\n"

    # Result of last action
    prompt += "The result of your last action, which might help inform your next action, is: \n"
    prompt += "```json\n"
    prompt += json.dumps(lastActionResult, indent=4, sort_keys=True)
    prompt += "```\n"

    # Format prompt
    prompt = "\n"
    prompt += "It's now time for you to choose an action.  Please provide a JSON dictionary with your action, below. "
    prompt += "The dictionary should have a key, `action`, and optionally up to two arguments, `arg1` and `arg2`, which are usually each UUIDs of specific objects, unless described differently in the action list.\n"

    # Dialog
    promptDialogStr = ""
    if (api.isAgentInDialog(agentIdx=0)):
        promptDialogStr = "*** NOTE: You are currently in a dialog.  Please ignore the above instructions about choosing an action, and instead choose which option you would like to say in the dialog. ***\n"
        promptDialogStr = "For reference, here is the dialog that you are currently in:\n"
        promptDialogStr += "```json\n"
        dialog = observation["ui"]["dialog_box"]
        promptDialogStr += json.dumps(dialog, indent=4, sort_keys=True)
        promptDialogStr += "```\n"
        promptDialogStr = "The expected response format is JSON, in between code brackets (```), as a dictionary with a single key: `chosen_dialog_option_int` (as well as `memory` and `running_hypothesis`).  The value should be an integer, corresponding to the dialog option you would like to select. You can write prose before the JSON code block, if that helps you think.\n"
    prompt += promptDialogStr


    ## Submit the prompt to the model
    response = submitLLMPrompt(prompt=prompt)   ## TODO: Implement this function

    ## Parse the response into valid JSON
    actionJSON = json.loads(response)
    ## The action JSON is often in the following format:
    # actionJSON = {
    #     "action": "USE",
    #     "arg1": OBJ_UUID1,
    #     "arg2": OBJ_UUID2
    # }

    # Return the action JSON
    return actionJSON



# Example of how to use the DiscoveryWorldAPI to interact with the environment
def exampleLoop(scenarioName, difficultyStr, randomSeed, maxSteps:int=10):
    print("Running example loop.")
    print("Task Theme: " + scenarioName)
    print("Difficulty: " + difficultyStr)
    print("Parametric Seed: " + str(randomSeed))
    print("\n")

    # Load the scenario
    api = DiscoveryWorldAPI(threadID=0)
    success = api.loadScenario(scenarioName = scenarioName, difficultyStr = difficultyStr, randomSeed = randomSeed, numUserAgents = 1)
    if (success == False):
        print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
        return None

    # Run a loop where we observe the environment, take an action, and repeat.
    lastActionResult = None
    for stepNum in range(0, maxSteps):
        # Print the result
        print("Step " + str(stepNum) + ":")

        # Get the observation from the environment
        observation = api.getAgentObservation(agentIdx=0)

        # TODO: Feed the observation into your model here, and have it generate an action
        actionJSON = model(api, observation, lastActionResult)
        print("Action: " + json.dumps(actionJSON))

        # Take the action
        result = api.performAgentAction(agentIdx=0, actionJSON=actionJSON)
        lastActionResult = result
        print("Result: " + json.dumps(result))
        print("\n")

        # Check if the task is complete
        if (api.areTasksComplete()):
            print("All tasks have been completed.  Exiting.")
            break

    # Show the final score
    scorecard = api.getTaskScorecard()
    print("Scorecard: ")
    print(json.dumps(scorecard, indent=4, sort_keys=True))


#
#   Main
#
if __name__ == "__main__":
    # Set the scenario and difficulty
    scenarioName = SCENARIOS[0]
    difficultyStr = SCENARIO_DIFFICULTY_OPTIONS[0]
    randomSeed = 0

    # Run the example loop
    exampleLoop(scenarioName=scenarioName, difficultyStr=difficultyStr, randomSeed=randomSeed, numSteps=10)