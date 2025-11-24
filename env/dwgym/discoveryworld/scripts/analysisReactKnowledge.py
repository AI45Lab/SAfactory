
import json
import os





# Main function
if __name__ == '__main__':


    # REACT
    #path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_easy/"
    #path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_challenge_azure/"
    path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_normal/"
    #path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_unit/"

    # Plan + Execute

    #path = "/home/peter/github/discoveryworld/paper-results/p+e/challenge/"
    #path = "/home/peter/github/discoveryworld/paper-results/p+e/normal/"
    #path = "/home/peter/github/discoveryworld/paper-results/p+e/easy/"
    #path = "/home/peter/github/discoveryworld/paper-results/p+e/unit/"

    print("Data path: " + str(path))

    # Get all files (recursively) in the directory
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            print(file)
            if ('_data.json' in file) and ("all_data.json" not in file):
                files.append(os.path.join(r, file))

    print("\n".join(files))


    out = []

    for file in files:
        # Look for the number of steps in the directory name (e.g. "-1000steps-")
        steps = 0
        fields = file.split("-")
        for field in fields:
            if "steps" in field:
                steps = int(field.split("steps")[0])
                break

        # Get just the filename (after the last "/")
        filename = file.split("/")[-1]

        # Remove the trailing "_data.json"
        justName = filename.split("_data.json")[0]

        fields = justName.split("_")
        taskName = fields[0]
        difficulty = fields[1]
        seed = int(fields[2])
        key = taskName + "_" + difficulty + "_" + str(steps) + "steps"

        # Load the data
        with open(file, 'r') as file1:
            data = json.load(file1)

        # Get the scorecard
        scorecard = data["metadata"]["final_scorecard"]
        print(type(scorecard))
        print(scorecard)
        if (type(scorecard) == list):
            numTasks = len(scorecard)
            if (numTasks > 1):
                print("WARNING: More than one task in scorecard")
                exit(1)
            scorecard = scorecard[0]

        normalizedScore = scorecard["scoreNormalized"]
        completedSuccessfully = scorecard["completedSuccessfully"]
        # Convert completedSuccessfully from boolean to int
        if (completedSuccessfully == True):
            completedSuccessfully = 1
        else:
            completedSuccessfully = 0


        packed = {
            "key": key,
            "taskName": taskName,
            "difficulty": difficulty,
            "seed": seed,
            "steps": steps,
            "normalizedScore": normalizedScore,
            "completedSuccessfully": completedSuccessfully
        }


        # ALSO, load the '_tracking.json' version of the file, which contains the actions, and the React agent's "thought"s
        filenameThoughts = file.replace("_data.json", "_tracking.jsonl")

        #with open(filenameThoughts, 'r') as file:
        #    dataThoughts = json.load(file)
        # Note, this one is a JSONL file, have to load it line-by-line
        allActions = []
        allThoughts = []
        print("Loading tracking information from: " + filenameThoughts)
        numLines = 0
        numNones = 0
        with open(filenameThoughts, 'r') as file1:
            for line in file1:
                #print("Line " + str(numLines) + ": " + line)
                dataLine = json.loads(line)
                #print("keys:")
                #print(dataLine.keys())
                # Look for "action" key
                if ("action" in dataLine):
                    allActions.append(dataLine["action"])
                    # Look for "thought" key
                    if (dataLine["action"] != None) and ("thought" in dataLine["action"]):
                        allThoughts.append(dataLine["action"]["thought"])
                numLines += 1

        print("Found " + str(len(allActions)) + " actions and " + str(len(allThoughts)) + " thoughts")
        packed["allActions"] = allActions
        packed["allThoughts"] = allThoughts


        # Compress all the thoughts together
        thoughtText = ""
        for idx, thought in enumerate(allThoughts):
            thoughtText += "Thought " + str(idx+1) + ": " + thought + "\n"
        print(thoughtText)

        packed["thoughtText"] = thoughtText

        # Try to run the knowledge scorer
        agentKnowledge = thoughtText
        knowledgeEvaluation = None
        if (agentKnowledge != None):
            from knowledgeScorer import KnowledgeScorer
            knowledgeScorer = KnowledgeScorer()
            # Example: evaluation = knowledgeScorer.evaluateKnowledge(scenarioName = "Space Sick", difficultyStr = "Easy", seed = 0, knowledgeToEvaluateStr=knowledgeToEvaluate)
            print("TaskName: " + taskName)
            print("Difficulty: " + difficulty)
            print("Seed: " + str(seed))
            print("Agent knowledge: ")
            #print(json.dumps(agentKnowledge, indent=4))    # as JSON
            print(agentKnowledge)   # as string
            knowledgeEvaluation = knowledgeScorer.evaluateKnowledge(scenarioName = taskName, difficultyStr = difficulty, seed = seed, knowledgeToEvaluateStr=agentKnowledge)
            print("Knowledge evaluation: " + str(knowledgeEvaluation))

        else:
            print("No agent knowledge found")

        knowledgeEvaluationScoreNormalized = None
        knowledgeEvaluationError = 0
        if (knowledgeEvaluation != None):
            knowledgeEvaluationScoreNormalized = knowledgeEvaluation[0]["evaluation_totalscore"]

        if (knowledgeEvaluationScoreNormalized == None):
            print("ERROR: No knowledge evaluation score found!")
            knowledgeEvaluationScoreNormalized = 0
            knowledgeEvaluationError = 1
            import time
            time.sleep(3)

        # Store
        packed["knowledgeEvaluationScoreNormalized"] = knowledgeEvaluationScoreNormalized
        packed["knowledgeEvaluationError"] = knowledgeEvaluationError
        packed["knowledgeEvaluation"] = knowledgeEvaluation

        #exit(1)
        out.append(packed)


    # Calculate averages across keys
    allKeys = set([x["key"] for x in out])
    averages = {}

    for key in allKeys:
        scores = [x["normalizedScore"] for x in out if x["key"] == key]
        completed = [x["completedSuccessfully"] for x in out if x["key"] == key]
        knowledgeScore = [x["knowledgeEvaluationScoreNormalized"] for x in out if x["key"] == key]
        knowledgeErrors = [x["knowledgeEvaluationError"] for x in out if x["key"] == key]

        avgScore = sum(scores) / len(scores)
        avgCompleted = sum(completed) / len(completed)
        avgKnowledgeScore = sum(knowledgeScore) / len(knowledgeScore)

        numSamples = len(scores)

        averages[key] = {
            "avgScore": avgScore,
            "avgCompleted": avgCompleted,
            "avgKnowledgeScore": avgKnowledgeScore,
            "numKnowledgeProcessingErrors": sum(knowledgeErrors),
            "numSamples": numSamples
        }


    # Sort both 'out' and 'averages' by key
    out = sorted(out, key=lambda x: x["key"])
    averages = {k: averages[k] for k in sorted(averages.keys())}

    # Write the data to a file
    packedOut = {
        "data": out,
        "averages": averages
    }

    print("\n\n")

    filenameOut = path + "performanceSummary.react.json"
    print("Writing " + filenameOut)
    with open(filenameOut, 'w') as file:
        json.dump(packedOut, file, indent=4)
    print("Done")


    # Load the data
    #with open('data.json', 'r') as file:
    #    data = json.load(file)