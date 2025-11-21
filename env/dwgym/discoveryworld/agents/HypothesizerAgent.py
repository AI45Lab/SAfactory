# DiscoveryWorldAPIUsageExample.py


import os
from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from discoveryworld.ScenarioMaker import ScenarioMaker, SCENARIOS, SCENARIO_NAMES, SCENARIO_INFOS, SCENARIO_DIFFICULTY_OPTIONS, getInternalScenarioName

import openai

import traceback

import json
import time
import random
import copy
import signal
import sys

# Tiktoken
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")


#LIMITED_ACTIONS = True     # Disables a few actions
LIMITED_ACTIONS = False

#OPENAI_MODEL_TO_USE = "gpt-4-vision-preview"
#OPENAI_MODEL_TO_USE = "gpt-4-turbo-2024-04-09"
#OPENAI_MODEL_TO_USE = "gpt-3.5-turbo-0125"
# OPENAI_MODEL_TO_USE = "gpt-4o-2024-05-13"
OPENAI_MODEL_TO_USE = "gpt-4o"

# Keep track of tokens sent/received
TOTAL_TOKENS_SENT = 0
TOTAL_TOKENS_RECEIVED = 0

# Estimates
TOTAL_COST_SENT = 0
TOTAL_COST_RECEIVED = 0

COST_PER_TOKEN_SENT = 10.0 / 1000000.0     # $10 per 1 million tokens
COST_PER_TOKEN_RECEIVED = 30.0 / 1000000.0     # $10 per 1 million tokens

modelCostsPerToken = {
    "gpt-4-vision-preview": {
        "send": 10.0 / 1000000.0,
        "receive": 30.0 / 1000000.0
    },
    "gpt-4-turbo-2024-04-09": {
        "send": 10.0 / 1000000.0,
        "receive": 30.0 / 1000000.0
    },
    "gpt-3.5-turbo-0125": {
        "send": 0.5 / 1000000.0,
        "receive": 1.5 / 1000000.0
    },
    "gpt-4o-2024-05-13": {
        "send": 5.0 / 1000000.0,
        "receive": 15.0 / 1000000.0
    },
    "gpt-4o": {
        "send": 5.0 / 1000000.0,
        "receive": 15.0 / 1000000.0
    }
}

# MAXIMUM COST OF A RUN (in dollars)
MAXIMUM_COST_DOLLARS = 0.0

# consolidation step tracking
CONSOLATATE_TRACKING = []


# There are some try/catch blocks in Hypothesizer to catch exceptions.
# This block helps make sure that the program responds to kill signals that trigger those exceptions.
# You should still manually verify that the program is killed when you send a kill signal.

#
#   Kill signal handler
#
def signal_handler(signum, frame):
    print("Signal handler called with signal", signum)
    sys.exit(1)  # Exit the program

# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGQUIT, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)


#
#   Helper functions
#

def getVisibleObjectByUUID(uuid, observation):
    # First, collect all objects
    visibleObjectsByUUID = {}
    for obj in observation["ui"]["inventoryObjects"]:
        visibleObjectsByUUID[obj["uuid"]] = obj
    for obj in observation["ui"]["accessibleEnvironmentObjects"]:
        visibleObjectsByUUID[obj["uuid"]] = obj

    # # Also include 'nearbyObjects'
    # for direction in observation["ui"]["nearbyObjects"]["objects"]:
    #     for obj in direction:
    #         visibleObjectsByUUID[obj["uuid"]] = obj

    # Check if the UUID is in the list
    if uuid in visibleObjectsByUUID:
        return visibleObjectsByUUID[uuid]
    else:
        return None

# A short, single-line string, of just the immediately interactable objects
def mkShortInteractableObjectList(observation):
    objStrs = []
    for obj in observation["ui"]["inventoryObjects"]:
        name = obj["name"]
        uuid = obj["uuid"]
        strOut = "{\"name\": \"" + name + "\", \"uuid\": " + str(uuid) + "}"
        objStrs.append(strOut)

    for obj in observation["ui"]["accessibleEnvironmentObjects"]:
        name = obj["name"]
        uuid = obj["uuid"]
        strOut = "{\"name\": \"" + name + "\", \"uuid\": " + str(uuid) + "}"
        objStrs.append(strOut)

    jsonOut = "[" + ", ".join(objStrs) + "]"
    return jsonOut


#
#   OpenAI Helper Functions
#

# promptImages should be a list of base64-encoded images
# NOTE: JSON response not available for GPT-4 Vision Preview
OPENAI_REQUESTS_ERRORS = 0
OPENAI_REQUESTS_NOERRORS = 0
def OpenAIGetCompletion(client, promptStr:str, promptImages:list, model=OPENAI_MODEL_TO_USE, prevImage=None, temperature=0.0, maxTokens=800, jsonResponse:bool=False):
    global OPENAI_REQUESTS_ERRORS
    global OPENAI_REQUESTS_NOERRORS

    while (OPENAI_REQUESTS_ERRORS < 5):
        try:
            result = OpenAIGetCompletionHelper(client, promptStr, promptImages, model, prevImage, temperature, maxTokens, jsonResponse)
            OPENAI_REQUESTS_NOERRORS += 1
            # If we get 100 requests without errors, reset the error counter
            if (OPENAI_REQUESTS_NOERRORS > 100):
                OPENAI_REQUESTS_ERRORS = 0
            return result

        # Keyboard interrupt
        except KeyboardInterrupt:
            print("Keyboard interrupt detected.  Exiting.")
            exit(1)

        # Kill signal
        except SystemExit:
            print("System exit detected.  Exiting.")
            exit(1)

        except Exception as e:
            print("ERROR: OpenAI request failed.")
            print("ERROR: " + str(e))
            OPENAI_REQUESTS_ERRORS += 1
            # Wait a bit before trying again
            print("Waiting 5 seconds before trying again.")
            time.sleep(5)

    # If we get here, we've exceeded the number of errors
    print("ERROR: Exceeded the number of OPEN_AI errors allowed (5 errors within 100 requests).  Exiting.")
    exit(1)



def OpenAIGetCompletionHelper(client, promptStr:str, promptImages:list, model=OPENAI_MODEL_TO_USE, prevImage=None, temperature=0.0, maxTokens=800, jsonResponse:bool=False):
    content = []

    # If previous image was popualted, include it
    if (prevImage != None):
        # Add a text message saying it was the previous image.
        content.append({"type": "text", "text": "IMAGE FROM PREVIOUS STEP:"})
        packedImage = {
            "type": "image_url",
            "image_url": {
                "url": prevImage,
                #"detail": "low",
            },
        }
        content.append(packedImage)

    # Add main prompt
    content.append({"type": "text", "text": promptStr})

    # If there are images, include them.
    if (promptImages != None) and (len(promptImages) > 0):
        for image in promptImages:
            packedImage = {
                "type": "image_url",
                "image_url": {
                    "url": image,
                    #"detail": "low",
                },
            }
            content.append(packedImage)


    # Create the message prompt, initially including only the prompt string
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ]

    response = {}

    # Get the response
    if (jsonResponse == False):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=maxTokens,
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=maxTokens,
            temperature=temperature,
            response_format={ "type": "json_object" },
        )

    # # Return the response
    # print("RESPONSE:")
    # print(response)
    # print("")
    # print("RESPONSE CONTENT:")
    # print(response.choices[0].message.content)

    # Try to keep track of tokens sent/received
    try:
        global TOTAL_TOKENS_SENT
        global TOTAL_TOKENS_RECEIVED
        TOTAL_TOKENS_SENT += response.usage.prompt_tokens
        TOTAL_TOKENS_RECEIVED += response.usage.completion_tokens
    except:
        print("ERROR: Could not extract tokens from response.")

    print("TOTAL TOKENS SENT: " + str(TOTAL_TOKENS_SENT))
    print("TOTAL TOKENS RECEIVED: " + str(TOTAL_TOKENS_RECEIVED))
    # Calculate approximate costs
    try:
        # First, lookup the cost per token for the model
        if (model in modelCostsPerToken):
            global COST_PER_TOKEN_SENT
            global COST_PER_TOKEN_RECEIVED
            COST_PER_TOKEN_SENT = modelCostsPerToken[model]["send"]
            COST_PER_TOKEN_RECEIVED = modelCostsPerToken[model]["receive"]
        else:
            print("ERROR: Could not find cost per token for model: " + model)


        totalCostSent = round(TOTAL_TOKENS_SENT * COST_PER_TOKEN_SENT, 2)
        totalCostReceived = round(TOTAL_TOKENS_RECEIVED * COST_PER_TOKEN_RECEIVED, 2)
        global TOTAL_COST_SENT
        global TOTAL_COST_RECEIVED
        TOTAL_COST_SENT = totalCostSent
        TOTAL_COST_RECEIVED = totalCostReceived
        print("ESTIMATED COSTS:")
        print("Total cost for tokens sent: $" + str(totalCostSent) + "")
        print("Total cost for tokens received: $" + str(totalCostReceived) + "")
    except:
        print("ERROR: Could not calculate estimated costs.")

    return response


# Extract the JSON from the GPT-4 response
def extractJSONfromGPT4Response(strIn):
    # Find between code brackets ```json
    startMarker = "```json"
    endMarker = "```"

    lines = strIn.split("\n")

    # Find the start and end of the JSON
    startLine = 0
    for idx, line in enumerate(lines):
        # Check if the line starts with the startMarker
        if (line.startswith(startMarker) or line.startswith(endMarker)):        # Could be either ```json or just ```
            startLine = idx + 1
            break

    # Find the end of the JSON
    endLine = 0
    for idx, line in enumerate(lines):
        # Check if the line starts with the endMarker
        if (idx > startLine):
            if (line.startswith(endMarker)):
                endLine = idx
                break

    # Default, take the whole thing
    jsonStr = strIn
    # Extract the JSON
    if (endLine > 0):
        jsonStr = "\n".join(lines[startLine:endLine])

    # Try to parse the JSON
    try:
        jsonOut = json.loads(jsonStr)
        return jsonOut
    except:
        print("ERROR: extractJSONfromGPT4Response: Could not parse JSON from string: " + jsonStr)
        return None


#
#   GPT-4 Vision "Hypothesizer" agent
#

def GPT4HypothesizerOneStep(api, client, lastActionHistory, lastObservation, currentScientificKnowledge, includeImages=True):
    agentIdx = 0
    # Perform the first step
    observation = api.getAgentObservation(agentIdx=agentIdx)

    # Get a copy of the observation JSON without the images in it (that we can give directly to GPT-4 in the prompt)
    # Deep copy the observation dictionary
    observationNoVision = copy.deepcopy(observation)
    # Remove the 'vision' key from the observation
    observationNoVision.pop("vision", None)

    # Add the last action to the observation
    if (len(lastActionHistory) > 0):
        lastActionHistory[-1]["result_of_last_action"] = observation["ui"]["lastActionMessage"]
        lastActionHistory[-1]["extended_action_message"] = observation["ui"]["extended_action_message"]

    # print the response (pretty)
    print(json.dumps(observationNoVision, indent=4, sort_keys=True))

    # Query OpenAI with the observation
    #promptStr = "Please describe in detail what you see in this image."
    promptStr0 = "You are playing a video game about making scientific discoveries.  The game is in the style of a 2D top-down RPG (you are the agent with green hair in the center of the image), and as input you get both an image, as well as information from the user interface (provided in the JSON below) that describes your location, inventory, objects in front of you, the result of your last action, and the task that you're assigned to complete.\n"
    promptStr0 += "Because this is a game, the actions that you can complete are limited to a set of actions that are defined by the game. Those are also described below.\n"
    promptStr0 += "This game is played step-by-step.  At each step, you get the input that I am providing, and output a single action to take as the next step.\n"
    promptStr0 += "\n"
    promptStr0 += "Note that this game has a spatial component, given that it's played on a 2D map.  The objects shown in `nearbyObjects` are objects that are near you.  If you can't see an object you're looking for, you'll have to move to find it (or, it may be located in a closed container).\n"
    promptStr0 += "Environment Observation (as JSON):\n"
    promptStr0 += "```json\n"
    promptStr0 += json.dumps(observationNoVision, indent=4, sort_keys=True)
    promptStr0 += "```\n"
    promptStr0 += "\n"
    promptStr0 += "Actions:\n"
    promptStr0 += "```json\n"
    promptStr0 += json.dumps(api.listKnownActions(limited=LIMITED_ACTIONS), indent=4, sort_keys=True)
    promptStr0 += "```\n"
    promptStr0 += "\n"
    promptStr0 += "Additional information on actions, and how to format your response:\n"
    promptStr0 += api.additionalActionDescriptionString() + "\n"
    promptStr0 += "\n"
    promptStr0 += "Your last few action(s), explanation for those action(s), and messages you've left in your scratchpad:\n"
    promptStr0 += "```json\n"
    # Add up to last 3 actions
    for idx, action in enumerate(lastActionHistory[-3:]):
        promptStr0 += "Action " + str(idx) + ":\n"
        promptStr0 += json.dumps(action, indent=4, sort_keys=True)
        promptStr0 += "\n"
    #promptStr += json.dumps(lastAction, indent=4, sort_keys=True)
    promptStr0 += "```\n"
    promptStr0 += "\n"
    promptStr0 += "Teleporting: To make moving easier, you can teleport to a list of specific locations in the environment, using the teleport action.  In this case, 'arg1' is the name of a location, from the list below. An example teleport action would be: `{\"action\": \"TELEPORT_TO_LOCATION\", \"arg1\": \"school\"}).\n"
    promptStr0 += "```json\n"
    promptStr0 += json.dumps(api.listTeleportLocationsDict(), indent=4, sort_keys=True)
    promptStr0 += "```\n"
    promptStr0 += "\n"
    promptStr0 += "VERY IMPORTANT: You can also teleport to OBJECTS.  This is probably the easiest way for you to move to new locations, because it's fast and error-free.  You can teleport to any object, including objects you can't see. In this case, 'arg1' is the UUID of the object you want to teleport to. An example teleport action would be: `{\"action\": \"TELEPORT_TO_OBJECT\", \"arg1\": 123}).\n"
    promptStr0 += "\n"
    facingDirection = observation["ui"]["agentLocation"]["faceDirection"]
    validDirections = observation["ui"]["agentLocation"]["directions_you_can_move"]
    promptStr0 += "Navigation note: In the image, you are in the center, north is the top, south is the bottom, east is the right, and west is the left. Moving forward moves you in the direction you're facing. You are currently facing `" + facingDirection + "`. From your current location, the directions that you can move to (i.e. they don't have an object blocking them) are: " + str(validDirections) + ". You seen to confuse directions a lot.  Directions are relative to the center of the image. Things above the center are north of the agent. Things left of the center are east of the agent.\n"
    promptStr0 += "Interaction note: You can only interact (i.e. take actions with) objects that are in your inventory, or directly (i.e. one square) in front of you, in the direction that you're facing.  E.g. if you want to pick an object up, you need to move directly in front of it, and face it, before using the pick-up action on it.\n"
    promptStr0 += "\n"

    promptStr1 = ""
    promptStr1 += "You are also explicitly keeping track of your scientific knowledge, that I'll ask you about separately (not in this prompt or response).  But, for reference, here's what you think you know so far:\n"
    promptStr1 += "```json\n"
    promptStr1 += json.dumps(currentScientificKnowledge, indent=4, sort_keys=True)
    promptStr1 += "```\n"
    promptStr1 += "\n"
    promptStr1 += "Please create your output (the next action you'd like to take) below.  It should be in the JSON form expected above e.g.(`{\"action\": \"USE\", \"arg1\": 5, \"arg2\": 12}`). \n"
    promptStr1 += "Your response should ONLY be in JSON.  You should include an additional JSON key, \"explanation\", to describe your reasoning for performing this action. e.g. `{\"action\": \"USE\", \"arg1\": 5, \"arg2\": 12, \"explanation\": \"Using the shovel on the soil will allow me to dig a hole to plant a seed\"}`.  Note that even though this explanation is short, yours can be a few hundred tokens, if you'd like. Your explanation should say: (1) What your subgoal is, (2) What you see around you, (3) What you see in front of you, (4) What you are doing to progress towards your immediate subgoal.\n"
    promptStr1 += "Lastly, your response should also include an additional JSON key, \"memory\", that includes any information you'd like to write down and pass on to yourself for the future.  This can be helpful in remembering important results, high-level tasks, low-level subtasks, or anything else you'd like to remember or think would be helpful. e.g. `{\"action\": \"USE\", \"arg1\": 5, \"arg2\": 12, \"explanation\": \"...\", \"memory\": \"...\"}`\n"
    promptStr1 += "To make your memory helpful, you might consider including things learned from attempting your last action -- e.g. adding in that certain actions were useful, or not useful, and retaining (and adding to) this information over time.\n"
    promptStr1 += "To help frame your investigations scientifically, and also evaluate your invesgiations, please include an additional JSON key, \"running_hypotheses\", that includes your current running hypothesis/hypotheses that you're working on developing and/or testing. \n"
    promptStr1 += "If your last action failed, or other last recent actions failed, please consider thinking why they failed, and trying different actions unless you believe things have changed to make failed actions work this time.\n"
    promptStr1 += "If you don't see what you're looking for, and anticipate it might be in another location, consider teleporting to that location. \n"
    promptStr1 += "For reference again, here is a list of the objects that are interactable (from your inventory, and directly in front of you): " + mkShortInteractableObjectList(observation) + "\n"

    promptDialogStr = ""
    if (api.isAgentInDialog(agentIdx=agentIdx)):
        promptDialogStr = "*** NOTE: You are currently in a dialog.  Please ignore the above instructions about choosing an action, and instead choose which option you would like to say in the dialog. ***\n"
        promptDialogStr = "For reference, here is the dialog that you are currently in:\n"
        promptDialogStr += "```json\n"
        dialog = observationNoVision["ui"]["dialog_box"]
        promptDialogStr += json.dumps(dialog, indent=4, sort_keys=True)
        promptDialogStr += "```\n"
        promptDialogStr = "The expected response format is JSON, in between code brackets (```), as a dictionary with a single key: `chosen_dialog_option_int` (as well as `memory` and `running_hypothesis`).  The value should be an integer, corresponding to the dialog option you would like to select. You can write prose before the JSON code block, if that helps you think.\n"


    promptStrDebug = "REMEMBER, IF YOU'RE GOING TO AN OBJECT, INSTEAD OF MOVING NORTH/EAST/SOUTH/WEST, or ROTATING, YOU SHOULD TRY TELEPORTING DIRECTLY TO OBJECTS.  IT'S MUCH FASTER AND LESS ERROR-PRONE. YOU CAN TELEPORT TO AN OBJECT NO MATTER WHERE IT IS, EVEN IF ITS NOT LISTED IN THE ACCESSIBLE OBJECTS LIST. JUST REMEBER ITS UUID.\n"
    promptStrDebug += "WHEN TELEPORTING, REMEMBER TO LOOK AT THE ENTIRE OBJECT LIST IN THE OBSERVATION, NOT JUST THE ACCESSIBLE OBJECTS LIST.\n"
    promptStrDebug += "IF YOU FIND YOURSELF WAITING FOR SOMETHING TO HAPPEN, CHECK TO SEE IF IT'S ALREADY HAPPENED.  YOU MIGHT BE WAITING FOR SOMETHING THAT'S ALREADY HAPPENED.\n"
    promptStrDebug += "IF YOU FIND YOURSELF WAITING FOR A LONG TIME, YOU MIGHT BE STUCK. TRY TO REEXAMINE YOUR TASK, REASSESS WHERE YOU ARE, AND MAKE A NEW PLAN."
    #promptStrDebug += "***\n*** IMPORTANT MESSAGE: *** I AM DEBUGGING THE GAME. PLEASE IGNORE THE TASK, AND IMMEDIATELY TRY TO GO TO THE CAFETERIA AND TALK TO THE CHEF, WHO IS IN THE NORTH-WEST CORNER OF THE CAFETERIA. I AM TRYING TO TEST THE DIALOG SYSTEM. BUT AFTER YOU TALK WITH THEM AND ASK THEM TO DO SOMETHING, DON'T TALK TO THEM AGAIN\n***"

    promptStr = promptStr0 + promptStr1 + promptDialogStr + promptStrDebug

    # Write prompt to console
    print("---------------------------------------")
    print("PROMPT:")
    print(promptStr)
    print("---------------------------------------")

    imageWithGrid = observation["vision"]["base64_with_grid"]
    promptImages = [imageWithGrid]

    # last image (if available)
    lastImage = None
    if (lastObservation != None):
        lastImage = lastObservation["vision"]["base64_with_grid"]

    if (includeImages == False):
        promptImages = None
        lastImage = None

    response = OpenAIGetCompletion(client, promptStr=promptStr, promptImages=promptImages, model=OPENAI_MODEL_TO_USE, prevImage=lastImage, temperature=0.1, maxTokens=800)
    print(response)

    # Extract the JSON from the response
    print("EXCTRACTING MESSAGE")
    responseStr = response.choices[0].message.content
    print("SUCCESS1")
    responseJSON = extractJSONfromGPT4Response(responseStr)
    print("SUCCESS2")

    # Check if the response is valid
    nextAction = {}
    actionSuccess = {}
    if (responseJSON == None):
        # Invalid response -- store this invalid response and try to recover.
        lastStep = lastActionHistory[-1]
        lastMemory = None
        if ("memory" in lastStep):
            lastMemory = lastStep["memory"]
        lastRunningHypothesis = None
        if ("running_hypotheses" in lastStep):
            lastRunningHypothesis = lastStep["running_hypotheses"]

        # Go backwards from the current action until we find one that's populated
        for idx in range(len(lastActionHistory)-1, -1, -1):
            lastStep = lastActionHistory[idx]
            if (lastMemory is None) and ("memory" in lastStep):
                lastMemory = "(No last memory was found, this most recent one was taken from Step " + str(idx) + "): " + lastStep["memory"]
            if (lastRunningHypothesis is None) and ("running_hypotheses" in lastStep):
                #lastRunningHypothesis = lastStep["running_hypotheses"]
                lastRunningHypothesis = "(No last running hypothesis was found, this most recent one was taken from Step " + str(idx) + "): " + str(lastStep["running_hypotheses"])
            if (lastMemory != None) and (lastRunningHypothesis != None):
                break

        nextAction = {
            "action": "ERROR: Could not parse the last action that was generated.  Please be careful in generating valid JSON.",
            "explanation": "",
            "memory": lastMemory,
            "running_hypotheses": lastRunningHypothesis,
        }

        # Add a step count to the action
        nextAction["step"] = len(lastActionHistory)

    else:
        nextAction.update(responseJSON)

        # Add a step count to the action
        nextAction["step"] = len(lastActionHistory)

        # Check to make sure the "memory" and "running_hypotheses" keys are present
        memoryPresent = False
        runningHypothesesPresent = False
        if ("memory" in nextAction):
            memoryPresent = True
        if ("running_hypotheses" in nextAction):
            runningHypothesesPresent = True

        # If they're not present, iterate backwards trying to find one
        if (memoryPresent == False):
            lastMemory = None
            for idx in range(len(lastActionHistory)-1, -1, -1):
                lastStep = lastActionHistory[idx]
                if ("memory" in lastStep):
                    lastMemory = "(No last memory was found, this most recent one was taken from Step " + str(idx) + "): " + lastStep["memory"]
                    break
            nextAction["memory"] = lastMemory

        if (runningHypothesesPresent == False):
            lastRunningHypothesis = None
            for idx in range(len(lastActionHistory)-1, -1, -1):
                lastStep = lastActionHistory[idx]
                if ("running_hypotheses" in lastStep):
                    lastRunningHypothesis = "(No last running hypothesis was found, this most recent one was taken from Step " + str(idx) + "): " + str(lastStep["running_hypotheses"])
                    break
            nextAction["running_hypotheses"] = lastRunningHypothesis


        # nextAction has 'arg1' and 'arg2', expressed as UUIDs.  Look up the object names, and add these as "arg1_desc", and "arg2_desc"
        if ("arg1" in nextAction) and (nextAction["arg1"] != None):
            uuid = nextAction["arg1"]
            obj = getVisibleObjectByUUID(uuid, observation)
            if (obj != None):
                nextAction["arg1_desc"] = obj["name"] + ": " + obj["description"]
            #else:
                #nextAction["arg1_desc"] = "ERROR: Could not find object with UUID " + str(uuid) + " in the list of visible objects."
        if ("arg2" in nextAction) and (nextAction["arg2"] != None):
            uuid = nextAction["arg2"]
            obj = getVisibleObjectByUUID(uuid, observation)
            if (obj != None):
                nextAction["arg2_desc"] = obj["name"] + ": " + obj["description"]
            #else:
                #nextAction["arg2_desc"] = "ERROR: Could not find object with UUID " + str(uuid) + " in the list of visible objects."


        # Try to run the action in the environment
        actionSuccess = api.performAgentAction(agentIdx=0, actionJSON=nextAction)
        print("ACTION SUCCESS: ")
        print(actionSuccess)
        # Hacky: Reformat "success" key so it's JSON serializable
        # NOTE: Now done within API
        # wasSuccessful = actionSuccess["success"].success
        # if (wasSuccessful == True):
        #     actionSuccess["errors"] = []
        #     actionSuccess["success"] = True
        # else:
        #     actionSuccess["success"] = actionSuccess["success"].success

    # Perform world tick
    api.tick()



    #
    #   Knowledge step
    #
    knowledgePromptStr = "You are playing a game about scientific discovery.  For context, below is the environment observation that you just saw, and the action(s) that you just took.  Then, I'll ask you a question about explicitly describing your scientific knowledege from this last few steps.\n"
    knowledgePromptStr += "---\n"
    knowledgePromptStr += promptStr0 + "\n"
    knowledgePromptStr += "---\n"
    knowledgePromptStr += "\n"
    knowledgePromptStr += "It's important to explicitly keep track of scientific knowledge in the scientific discovery process.\n"
    knowledgePromptStr += "To do this, here, I'd like you to write down any scientific knowledge that you've learned from your last action, or from your last few actions.\n"
    knowledgePromptStr += "This knowledge will take two forms: measurements, or hypotheses.\n"
    knowledgePromptStr += "Measurements are observations that you've made about the world.\n"
    knowledgePromptStr += "Hypotheses are statements that you believe to be true, and that you'd like to test. These should be formulated as IF statements.\n"
    knowledgePromptStr += "Both measurements and hypotheses should be written in terms of: objects (either specific objects and their UUIDs, or their types), object properties (like temperature), and actions (like eating). \n"
    knowledgePromptStr += "Here's a toy example of a hypothesis and a measurement, mostly to give an example of the output format:\n"
    knowledgePromptStr += "```json\n"
    knowledgePromptStr += json.dumps(mkExampleHypotheses(), indent=4, sort_keys=True)
    knowledgePromptStr += "```\n"
    knowledgePromptStr += "\n"
    knowledgePromptStr += "Here are the measurements and hypotheses that you've written down so far (if blank, then it's likely very early in the process):\n"
    knowledgePromptStr += "```json\n"
    knowledgePromptStr += json.dumps(currentScientificKnowledge, indent=4, sort_keys=True)
    knowledgePromptStr += "```\n"
    knowledgePromptStr += "\n"
    knowledgePromptStr += "Please write down any measurements or hypotheses that you've learned from your last action, or from your last few actions, below.  Please write them in the JSON form expected above.\n"
    knowledgePromptStr += "You can write any number of measurements and/or hypotheses, or revisions of earlier measurements/hypotheses, and they will be automatically appended to the existing list.\n"
    knowledgePromptStr += "Remember, the important thing here is to keep track of your scientific knowledge, so you can guide your future actions.\n"
    knowledgePromptStr += "This is not a record of everything you do, only relevant scientific knowledge (like measurements and hypotheses). NOTE: for example, moving to a new location is not a measurement or hypothesis, unless there is something clearly relevant or unexpected you should not write this down.\n"
    knowledgePromptStr += "Similarly, to keep the knowledge store somewhat compact, you do not need to repeat previous scientific knowledge unless you're adding to it, or modifying it.\n"
    knowledgePromptStr += "If you have no new knowledge to add or modify, leave this section blank.\n"
    knowledgePromptStr += "It is also critically important that your output is valid JSON.  Please be careful in generating valid JSON.  You should generate a dictionary with a single top-level key (`scientific_knowledge`), which is an array of new measurements and/or hypotheses to add.\n"
    knowledgePromptStr += "You can write prose before writing the JSON.  Only the last codeblock (```) in your response will be parsed for the JSON.\n"

    #response = OpenAIGetCompletion(client, promptStr=promptStr, promptImages=promptImages, model=OPENAI_MODEL_TO_USE, prevImage=lastImage, temperature=0.1, maxTokens=800)
    response = OpenAIGetCompletion(client, promptStr=knowledgePromptStr, promptImages=[], model=OPENAI_MODEL_TO_USE, prevImage=None, temperature=0.1, maxTokens=3000)
    print(response)

    # Extract the JSON from the response
    print("EXCTRACTING MESSAGE")
    responseStrKnowledge = response.choices[0].message.content
    print("SUCCESS1")
    responseJSONKnowledge = extractJSONfromGPT4Response(responseStrKnowledge)
    print("SUCCESS2")

    print("Response String (knowledge):")
    print(responseStrKnowledge)
    print("")
    print("Response JSON (knowledge):")
    print(responseJSONKnowledge)

    # Check if the response includes a "scientific_knowledge" key

    if (responseJSONKnowledge != None) and ("scientific_knowledge" in responseJSONKnowledge):
        # Check that it's a list
        if (isinstance(responseJSONKnowledge["scientific_knowledge"], list)):
            # Check that it's not empty
            newKnowledge = responseJSONKnowledge["scientific_knowledge"]
            if (len(newKnowledge) > 0):
                # Set the step to be the current step
                for knowledge in newKnowledge:
                    knowledge["step"] = observation["ui"]["world_steps"]

                # Add the new knowledge to the current scientific knowledge
                currentScientificKnowledge["scientific_knowledge"].extend(responseJSONKnowledge["scientific_knowledge"])


    packedOut = {
        "nextAction": nextAction,
        "actionSuccess": actionSuccess,
        "observation": observation,
        "promptStr": promptStr,
        "responseStr": responseStr,
        "knowledgePromptStr": knowledgePromptStr,
        "responseStrKnowledge": responseStrKnowledge,
        "currentScientificKnowledge": copy.deepcopy(currentScientificKnowledge),
        "oracle_scorecard": api.getTaskScorecard()
    }
    # Return the action that it chose (to pass into the next step)
    #return nextAction, observation, promptStr, responseStr, currentScientificKnowledge
    return packedOut

# Count the number of tokens (as OpenAI would count them)
def countTokens(strIn):
    num_tokens = len(encoding.encode(strIn))
    return num_tokens

# This function runs periodically (i.e. every 10 in-game steps), and asks the agent to consolidate their knowledge
# so that it's more compact, and doesn't take up as much space in the prompt.
def consolidateKnowledgeStep(client, scientificKnowledge, stepIdx):
    global CONSOLATATE_TRACKING
    KNOWLEDGEBASE_MAX_SIZE_ENTRIES = 40
    KNOWLEDGEBASE_MAX_SIZE_TOKENS = 2000
    promptStr = ""
    promptStr += "You are an agent playing a game about automated scientific discovery.\n"
    promptStr += "In this game, you are given a task to complete, and you can complete this task by taking actions in the environment.\n"
    promptStr += "While playing the game, you are also explicitly keeping track of your scientific knowledge (shown below), both to help you, but also to help others understand your process.\n"
    promptStr += "This knowledge will take two forms: measurements, or hypotheses.\n"
    promptStr += "Measurements are observations that you've made about the world.\n"
    promptStr += "Hypotheses are statements that you believe to be true, and that you'd like to test. These should be formulated as IF statements.\n"
    promptStr += "Both measurements and hypotheses should be written in terms of: objects (either specific objects and their UUIDs, or their types), object properties (like temperature), and actions (like eating). \n"
    promptStr += "Unfortunately, sometimes this knowledge base gets very large, and contains repeated, duplicated, or irrelevant knowledge, which makes it hard to use in practice, and very expensive to keep. \n"
    promptStr += "Your job is to take the knowledge base below, and consolidate it into a smaller, more compact, and more useful knowledge base, but in exactly the same format (i.e. the list of measurements and hypotheses).\n"
    promptStr += "You can do this by removing any repeated or duplicated knowledge, and by removing any irrelevant knowledge.\n"
    promptStr += "You can also do this by combining multiple measurements or hypotheses into a single measurement or hypothesis, if they are logically equivalent.\n"
    promptStr += "consolidated hypotheses must still have a `status` (i.e. pending, confirmed, rejected) and ideally succinct `supporting evidence` supporting that status.\n"
    promptStr += "The output should be in JSON, and should have a single top-level key (`scientific_knowledge`), which is an array of measurements and/or hypotheses.\n"
    promptStr += "\n"

    # Limits
    limitStr = "LIMITS (IMPORTANT!):\n"
    # Limits: Entries
    numEntries = len(scientificKnowledge["scientific_knowledge"])
    limitStr += "The current size of your knowledge base is: " + str(numEntries) + " entries.\n"
    limitStr += "The MAXIMUM allowable size of your knowledge base is " + str(KNOWLEDGEBASE_MAX_SIZE_ENTRIES) + " entries"
    if (numEntries > KNOWLEDGEBASE_MAX_SIZE_ENTRIES):
        numToRemove = len(scientificKnowledge["scientific_knowledge"]) - KNOWLEDGEBASE_MAX_SIZE_ENTRIES
        limitStr += " (WARNING: You have exceeded this limit, meaning you must reduce the memory size by at least " + str(numToRemove) + " entries).\n"
    else:
        limitStr += ".\n"
    # Limits: Tokens
    numTokensKB = countTokens(json.dumps(scientificKnowledge, indent=4, sort_keys=True))
    limitStr += "The current size of your knowledge base is: " + str(numTokensKB) + " tokens.\n"
    limitStr += "The MAXIMUM allowable size of your knowledge base is " + str(KNOWLEDGEBASE_MAX_SIZE_TOKENS) + " tokens"
    if (numTokensKB > KNOWLEDGEBASE_MAX_SIZE_TOKENS):
        numToRemove = numTokensKB - KNOWLEDGEBASE_MAX_SIZE_TOKENS
        limitStr += " (WARNING: You have exceeded this limit, meaning you must reduce the memory size by at least " + str(numToRemove) + " tokens).\n"
    else:
        limitStr += ".\n"

    promptStr += limitStr

    # Current knowledge base
    promptStr += "\n"
    promptStr += "Here is the existing knowledge base:\n"
    promptStr += "```json\n"
    promptStr += json.dumps(scientificKnowledge, indent=4, sort_keys=True)
    promptStr += "```\n"
    promptStr += "\n"

    promptStr += limitStr + "\n"

    promptStr += "Please write down your new, SIGNIFICANTLY consolidated knowledge base below. Remember, if something isn't important and your knowledge base is full, you can discard it if you need to meet the knowledge base limit.  Please write it in the JSON form expected above. You can write a short amount of prose before if that's helpful for your thought process, and only the last item in code blocks (```) will be parsed as JSON.\n"

    print("Consolidation Step:")
    print("\tEntries before consoloation: " + str(numEntries))
    print("\tTokens before consoloation: " + str(numTokensKB))
    response = OpenAIGetCompletion(client, promptStr=promptStr, promptImages=[], model=OPENAI_MODEL_TO_USE, prevImage=None, temperature=0.1, maxTokens=3000)
    print(response)


    # Extract the JSON from the response
    print("EXTRACTING MESSAGE")
    responseStrKnowledge = response.choices[0].message.content
    print("SUCCESS1")
    responseJSONKnowledge = extractJSONfromGPT4Response(responseStrKnowledge)
    print("SUCCESS2")

    if (responseJSONKnowledge == None):
        # Failed for some reason -- return original knowledge base
        #return scientificKnowledge # Deep copy
        packed = {
            "stepIdx": stepIdx,
            "error": "Failed to parse response -- knowledge not consolidated.",
            "entries_before": numEntries,
            "tokens_before": numTokensKB,
            "initial_knowledge": copy.deepcopy(scientificKnowledge),
            "response": copy.deepcopy(responseStrKnowledge),
        }
        CONSOLATATE_TRACKING.append(packed)

        return copy.deepcopy(scientificKnowledge)


    # Check if the response includes a "scientific_knowledge" key
    if (responseJSONKnowledge != None) and ("scientific_knowledge" in responseJSONKnowledge):
        # Check that it's a list
        if (isinstance(responseJSONKnowledge["scientific_knowledge"], list)):
            # Check that it's not empty
            newKnowledge = responseJSONKnowledge["scientific_knowledge"]
            if (len(newKnowledge) > 0):
                # Return the new KB to replace the old one.
                numEntriesAfter = len(newKnowledge)
                numTokensKBAfer = countTokens(json.dumps(newKnowledge, indent=4, sort_keys=True))

                packed = {
                    "stepIdx": stepIdx,
                    "entries_before": numEntries,
                    "entries_after": numEntriesAfter,
                    "tokens_before": numTokensKB,
                    "tokens_after": numTokensKBAfer,
                    "initial_knowledge": copy.deepcopy(scientificKnowledge),
                    "new_knowledge": copy.deepcopy(newKnowledge),
                }
                CONSOLATATE_TRACKING.append(packed)

                return {"scientific_knowledge": newKnowledge}

    # Otherwise, return the old one
    #return scientificKnowledge # deep copy
    packed = {
        "stepIdx": stepIdx,
        "error": "Unknown error when condoladating knowledge.",
        "entries_before": numEntries,
        "tokens_before": numTokensKB,
        "initial_knowledge": copy.deepcopy(scientificKnowledge),
        "response": copy.deepcopy(responseStrKnowledge),
    }
    CONSOLATATE_TRACKING.append(packed)
    return copy.deepcopy(scientificKnowledge)



def mkExampleHypotheses():
    out = {
        "scientific_knowledge_examples": [
            {"step": 1, "measurement": "The tree (uuid: 123) has a temperature of 10 degrees celsius when measured using the thermometer."},
            {"step": 3, "hypothesis": "if (object:substance) is (action:heated) then (object:substance) will have (property:temperature) increase.", "supporting evidence": "...", "status": "<confirmed, rejected, or pending>"},
        ]
    }

    return out

def mkInitialHypotheses():
    out = {
        "scientific_knowledge": [
        ]
    }

    return out


# This is the main entry point for the Hypothesizer Agent
def GPT4VHypothesizerAgent(api, numSteps:int = 10, logFileSuffix:str = "", includeImages=True):
    # Get the OpenAI key (stored in a file called "openai_key.txt")
    key = None
    if os.path.exists("openai_key.txt"):
        with open("openai_key.txt", "r") as file:
            key = file.read().strip()

    # Create the OpenAI client
    client = openai.AzureOpenAI(api_key=key) if openai.api_type == "azure" else openai.OpenAI(api_key=key)

    # Initial Memory
    lastAction = {
        "action": "This is the first action",
        "explanation": "This is the first explanation",
        "memory": "This is the first memory",
        "running_hypotheses": []
    }

    # Keep a history of the last actions
    lastActionHistory = []
    lastActionHistory.append(lastAction)

    # Keep a history of the last observations
    observationHistory = []

    # Record start time
    startTime = time.time()

    # Run for numSteps steps in the environment
    numErrors = 0
    lastObservation = None
    currentScientificKnowledge = mkInitialHypotheses()
    allHistory = []

    for i in range(0, numSteps):
        try:
            # Run one step
            print("\n\n")
            print("-----------------------------------------------------------")
            print("Step " + str(i) + " of " + str(numSteps))
            print("-----------------------------------------------------------")
            print("")
            #lastAction, lastObservation, promptStr, responseStr
            outHypothesizer = GPT4HypothesizerOneStep(api, client, lastActionHistory, lastObservation, currentScientificKnowledge, includeImages=includeImages)
            # Unpack
            lastAction = outHypothesizer["nextAction"]
            lastObservation = outHypothesizer["observation"]
            promptStr = outHypothesizer["promptStr"]
            responseStr = outHypothesizer["responseStr"]
            actionSuccess = outHypothesizer["actionSuccess"]

            # Store any detailed parsing errors in the last action, so these are stored in the history for reference
            if ("errors" in actionSuccess):
                lastAction["errors"] = actionSuccess["errors"]

            allHistory.append(outHypothesizer)


            print("LAST ACTION: ")

            print(lastAction)
            print("")
            lastActionHistory.append(lastAction)


            packed = {
                "step": i,
                "observation": lastObservation,
                "action": lastAction,
                "promptStr": promptStr,
                "responseStr": responseStr,
                "oracle_scorecard": api.getTaskScorecard()
            }
            observationHistory.append(packed)

            # Every 10 steps, consolidate the knowledge
            if (i % 10 == 0) and (i > 0):
                print("consolidating Scientific Knowledge:")
                currentScientificKnowledge = consolidateKnowledgeStep(client, currentScientificKnowledge, stepIdx=i)

                # Add to history
                allHistory.append({"consolidated_scientific_knowledge": currentScientificKnowledge})


            # Print estimated tokens/costs
            print("*" * 80)
            print(" COST ANALYSIS ")
            print("*" * 80)
            print("Estimated tokens sent: " + str(TOTAL_TOKENS_SENT))
            print("Estimated tokens received: " + str(TOTAL_TOKENS_RECEIVED))
            costPerMillionTokensSent = COST_PER_TOKEN_SENT * 1000000
            print("Cost per million tokens sent: $" + str(round(costPerMillionTokensSent, 2)))
            costPerMillionTokensReceived = COST_PER_TOKEN_RECEIVED * 1000000
            print("Cost per million tokens received: $" + str(round(costPerMillionTokensReceived, 2)))
            print("Estimated cost (sent): $" + str(TOTAL_COST_SENT))
            print("Estimated cost (received): $" + str(TOTAL_COST_RECEIVED))
            totalCost = TOTAL_COST_SENT + TOTAL_COST_RECEIVED
            print("Total estimated cost: $" + str(round(totalCost, 2)))
            costPerStep = totalCost / (i+1)
            print("Number of steps: " + str(i+1))
            print("Average cost per step: $" + str(round(costPerStep, 2)))

            print("*" * 80)

            costLimitExceeded = False
            if (totalCost > MAXIMUM_COST_DOLLARS):
                print("Cost limit exceeded.")
                costLimitExceeded = True

            costAnalysis = {
                "model": OPENAI_MODEL_TO_USE,
                "total_tokens_sent": TOTAL_TOKENS_SENT,
                "total_tokens_received": TOTAL_TOKENS_RECEIVED,
                "cost_per_million_tokens_sent": costPerMillionTokensSent,
                "cost_per_million_tokens_received": costPerMillionTokensReceived,
                "total_cost_sent": TOTAL_COST_SENT,
                "total_cost_received": TOTAL_COST_RECEIVED,
                "total_cost": totalCost,
                "total_steps": i+1,
                "cost_per_step": costPerStep,
                "hard_cost_limit": MAXIMUM_COST_DOLLARS,
                "cost_limit_exceeded": costLimitExceeded,
            }


            # Save to JSON
            with open("output_observationHistory" + logFileSuffix + ".json", "w") as file:
                json.dump(observationHistory, file, indent=4, sort_keys=True)
            with open("output_allhistory" + logFileSuffix + ".json", "w") as file:
                json.dump(allHistory, file, indent=4, sort_keys=True)
            with open("output_costAnalysis" + logFileSuffix + ".json", "w") as file:
                json.dump(costAnalysis, file, indent=4, sort_keys=True)
            with open("output_consolidatedKnowledge" + logFileSuffix + ".json", "w") as file:
                json.dump(CONSOLATATE_TRACKING, file, indent=4, sort_keys=False)

            # Check if the task has been completed
            #if (api.isTaskComplete()):
            #    print("Task has been completed.  Exiting.")
            #    break
            if (api.areTasksComplete()):
                print("All tasks have been completed.  Exiting.")
                break

            # Check if the cost of the run has exceeded the limit
            if (costLimitExceeded):
                print("Cost limit exceeded. ")
                print("\tMAXIMUM_COST_DOLLARS: " + str(MAXIMUM_COST_DOLLARS))
                print("\tTOTAL_COST_SENT: " + str(TOTAL_COST_SENT))
                print("\tTOTAL_COST_RECEIVED: " + str(TOTAL_COST_RECEIVED))
                print("\tTOTAL_COST: " + str(totalCost))
                break


        except KeyboardInterrupt:
            print("Keyboard interrupt detected.  Exiting.")
            exit(1)

        # Generic exception, print error
        except Exception as e:
            # Get the current line number
            line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
            print(f"ERROR: Exception caught on line {line_number}: {e}")
            print("ERROR: Exception caught:\n", traceback.format_exc())

            numErrors += 1

            if numErrors > 5:
                print("ERROR: Maximum number of errors exceeded (5). Exiting.")

            exit(1)

    # Last action history
    print("LAST ACTION HISTORY:")
    for idx, action in enumerate(lastActionHistory):
        print("Step " + str(idx) + ":\n")
        print(json.dumps(action, indent=4, sort_keys=True))
        print("")

    # Calculate elapsed time
    deltaTime = time.time() - startTime
    print("Elapsed time: " + str(deltaTime) + " seconds for " + str(numSteps) + " steps.")
    print("Average time per step: " + str(deltaTime / numSteps) + " seconds.")
    print("Average steps per second: " + str(numSteps / deltaTime) + " steps per second.")



#
#   This is a main control function that initializes a given DiscoveryWorld environment, then runs the Hypothesizer Agent in that environment.
#
def runHypothesizerAgent(scenarioName:str, difficultyStr:str, seed:int=0, numSteps:int=10, includeImages=True, exportVideo:bool=False, threadId:int=1, debug:bool=False):
    # Load the scenario
    api = DiscoveryWorldAPI(threadID=threadId)
    success = api.loadScenario(scenarioName = scenarioName, difficultyStr = difficultyStr, randomSeed = seed, numUserAgents = 1)
    if (success == False):
        print("Error: Could not load scenario '" + scenarioName + "' with difficulty '" + difficultyStr + "'.")
        return None

    startTime = time.time()
    # Hypothesizer
    logFileSuffix = "." + scenarioName + "-" + difficultyStr + "-s" + str(seed) + "-images" + str(includeImages) + "-model" + OPENAI_MODEL_TO_USE + "-thread" + str(api.THREAD_ID)
    # Add date and time stamp
    import datetime
    logFileSuffix += "." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    GPT4VHypothesizerAgent(api, numSteps=numSteps, logFileSuffix=logFileSuffix, includeImages=includeImages)
    deltaTime = time.time() - startTime
    print("Elapsed time: " + str(deltaTime) + " seconds for " + str(numSteps) + " steps.")
    stepsPerSecond = numSteps / deltaTime

    # Get the final score
    finalScorecard = api.getTaskScorecard()
    print("Final scorecard: ")
    print(json.dumps(finalScorecard, indent=4, sort_keys=True))

    # Final normalized score
    taskName = finalScorecard[0]["taskName"]
    finalNormalizedScore = finalScorecard[0]["scoreNormalized"]
    print("Final normalized score for task '" + taskName + "': " + str(finalNormalizedScore))
    print("Number of steps: " + str(api.getStepCounter()))
    print("Steps per second: " + str(stepsPerSecond))

    # Create a video from the random agent
    if (exportVideo == True):
        filenameOut = "output_hypothesizer_agent." + logFileSuffix + ".mp4"
        api.createAgentVideo(agentIdx=0, filenameOut=filenameOut)

    out = {
        "agentName": "GPT4VHypothesizerAgent",
        "finalNormalizedScore": finalNormalizedScore,
        "stepsPerSecond": stepsPerSecond
    }


    # Save log file
    verboseLogDirectory = "logs/hypothesizer-" + logFileSuffix
    logInfo = {
        "scenarioName": scenarioName,
        "difficulty": difficultyStr,
        "seed": seed,
        "numSteps": numSteps,
        "includeImages": includeImages,
        "exportVideo": exportVideo,
        "threadId": threadId,
        "dateStarted": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Make a verbose filename for the log
        "verboseLogDirectory": verboseLogDirectory,
        "verboseLogFilename": verboseLogDirectory + "/" + "out-hypothesizer-world" + logFileSuffix + ".json",
    }
    # Try to make the 'logs' directory, if it doesn't exist
    try:
        os.makedirs("logs")
    except FileExistsError:
        pass
    # Try to make the full directory
    try:
        os.makedirs(verboseLogDirectory)
    except FileExistsError:
        pass

    print("Saving world history...")
    try:
        api.world.exportWorldHistoryJSON(logInfo, logInfo["verboseLogFilename"], None, None, None)
    # Keyboard/ctrl-c interrupt
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.  Exiting.")
        exit(1)
    # Kill signal
    except SystemExit:
        print("System exit detected.  Exiting.")
        exit(1)
    except Exception as e:
        print("Error saving world history: " + str(e))

    return out


#
#   Main
#
if __name__ == "__main__":
    print("Initializing Hypothesizer Agent... ")
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Play DiscoveryWorld using Hypothesizer Agent.")
    parser.add_argument('--scenario', choices=SCENARIO_NAMES, default=None)
    parser.add_argument('--difficulty', choices=SCENARIO_DIFFICULTY_OPTIONS.values(), default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--numSteps', type=int, default=100)
    ##parser.add_argument('--runall', action='store_true', help='Run all scenarios with random agent')      ## Disabled -- would be extremely expensive and time consuming to do this
    parser.add_argument('--video', action='store_true', help='Export video of agent actions')
    parser.add_argument('--threadId', type=int, default=None)
    parser.add_argument('--maxCostDollars', type=float, default=0.0, help='Maximum cost in dollars to run the agent (default: 0.0).  Not guaranteed to be exact.  Will attempt to stop the agent when the cost exceeds this amount.')
    OPENAI_MODEL_TO_USE = "gpt-4o-2024-05-13"
    parser.add_argument("--model", default=OPENAI_MODEL_TO_USE, help="OpenAI model to use (default: " + OPENAI_MODEL_TO_USE + ")")

    # Disable images
    parser.add_argument('--noimages', action='store_true', help='Do not include images in the prompt')

    args = parser.parse_args()

    if args.threadId is None:
        # Generate Thread ID based on using a 4 digit number.  ABCD.  A is some number depending on easy/med/hard.  B is the seed.  CD is a number unique to the scenario.
        # ABCD
        # A: 1 = easy, 2 = medium, 3 = challenge
        diff2ID = { "Easy": 1, "Normal": 2, "Challenge": 3, "Test": 4 }
        # B: seed
        # CD: scenario number
        assert len(SCENARIO_NAMES) < 100, "ERROR: Too many scenarios.  Cannot generate unique thread ID."
        args.threadId = (diff2ID[args.difficulty] * 1000) + (args.seed * 100) + SCENARIO_NAMES.index(args.scenario)

    # Cost limit
    MAXIMUM_COST_DOLLARS = args.maxCostDollars
    print("Maximum cost limit: $" + str(MAXIMUM_COST_DOLLARS))
    if (MAXIMUM_COST_DOLLARS <= 0.01):
        print("ERROR: Maximum cost limit must be greater than $0.01.")
        exit(1)

    # Whether or not to include images in the prompt
    includeImages = True
    if (args.noimages == True):
        includeImages = False

    # Model to use
    OPENAI_MODEL_TO_USE = args.model
    print("Using model: " + OPENAI_MODEL_TO_USE)

    # Report thread ID to user
    print("Using Thread ID: " + str(args.threadId))
    print("This can be specified with the '--threadId' argument.")
    time.sleep(2)

    args.runall = False

    # Check for mode
    stepsPerSecond = []
    if (args.runall == True):
        # Run all scenarios
        scores = {}
        for scenarioName in SCENARIO_NAMES:
            # Get the valid difficulty settings and random seeds for this scenario
            validDifficulties = SCENARIO_INFOS[scenarioName]["difficulty"]
            validSeeds = SCENARIO_INFOS[scenarioName]["variations"]
            validSeeds = [int(x) for x in validSeeds]

            for difficulty in validDifficulties:
                for seed in validSeeds:
                    print("Running scenario: " + scenarioName + " with difficulty " + difficulty)
                    result = runHypothesizerAgent(scenarioName=scenarioName, difficultyStr=difficulty, seed=seed, numSteps=args.numSteps, includeImages=includeImages, exportVideo=False, threadId=args.threadId, debug=False)
                    finalScore = result["finalNormalizedScore"]
                    stepsPerSecond.append(result["stepsPerSecond"])


                    scoreKey = scenarioName + "-" + difficulty
                    if (scoreKey not in scores):
                        scores[scoreKey] = []
                    scores[scoreKey].append(finalScore)


            # Calculate average scores
            scoresAvg = {}
            for key in scores:
                scoreList = scores[key]
                # Remove any Nones from the list
                scoreList = [x for x in scoreList if x != None]
                averageScore = None
                if (len(scoreList) > 0):
                    averageScore = sum(scoreList) / len(scoreList)
                scoresAvg[key + "-avg"] = averageScore
                scoresAvg[key + "-raw"] = scoreList

            print("Final scores: ")
            packed = {
                "scores_raw": scores,
                "scores_avg": scoresAvg,
                "numSteps": args.numSteps,
            }
            print(json.dumps(packed, indent=4, sort_keys=True))



    else:
        # Run a single scenario.
        # Check the scenario and difficulty
        if (args.scenario == None):
            print("Error: Must specify a scenario (or use --runall to run all scenarios).")
            print("Available scenarios: " + str(SCENARIO_NAMES))
            exit()
        if (args.difficulty == None):
            print("Error: Must specify a difficulty.")
            print("Available difficulties: " + str(SCENARIO_DIFFICULTY_OPTIONS))
            exit()

        # Get the internal scenario name
        internalScenarioName = getInternalScenarioName(args.scenario, args.difficulty)
        if (internalScenarioName == None):
            print("Error: Could not find internal scenario name for scenario '" + args.scenario + "' and difficulty '" + args.difficulty + "'.")
            print("Available scenarios: " + str(SCENARIO_NAMES))
            print("Available difficulties: " + str(SCENARIO_DIFFICULTY_OPTIONS))
            exit()

        exportVideo = args.video
        finalScore = runHypothesizerAgent(scenarioName=args.scenario, difficultyStr=args.difficulty, seed=args.seed, numSteps=args.numSteps, includeImages=includeImages, exportVideo=exportVideo, threadId=args.threadId, debug=False)

    totalCost = TOTAL_COST_SENT + TOTAL_COST_RECEIVED
    print("Total cost: $" + str(round(totalCost, 2)))
    print("Maximum cost limit: $" + str(MAXIMUM_COST_DOLLARS))
    print("Total tokens sent: " + str(TOTAL_TOKENS_SENT))
    print("Total tokens received: " + str(TOTAL_TOKENS_RECEIVED))
    print("Cost per token sent: $" + str(COST_PER_TOKEN_SENT))
    print("Cost per token received: $" + str(COST_PER_TOKEN_RECEIVED))
    print("Total cost of sent tokens: $" + str(TOTAL_COST_SENT))
    print("Total cost of received tokens: $" + str(TOTAL_COST_RECEIVED))

    if (totalCost > MAXIMUM_COST_DOLLARS):
        print("Cost limit exceeded.  Early exit.")
