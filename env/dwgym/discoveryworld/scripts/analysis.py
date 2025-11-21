# Analysis.py

import json
import os


# Load all the JSON files from a given path
def loadDataJSON(path:str):
    filterPrefix = "discoveryworld-playlog"

    jsonFiles = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    # Filter out the files that don't start with the prefix
    jsonFiles = [file for file in jsonFiles if file.startswith(filterPrefix)]

    # The files will be of the form "<filename>.partXofY.json" -- Try to parse out the number of parts
    # and sort the files accordingly
    jsonFiles.sort()
    numParts = 0
    if len(jsonFiles) > 0:
        numParts = int(jsonFiles[-1].split(".part")[1].split("of")[1].split(".json")[0])

    print("Number of parts: " + str(numParts))

    # Get the part of the filename before the ".partXofY.json" part
    filePrefix = jsonFiles[0].split(".part")[0]

    # Load each file, one by one
    worldHistory = []
    for i in range(numParts):
        filename = filePrefix + ".part" + str(i) + "of" + str(numParts) + ".json"
        print("Loading file: " + filename)
        with open(path + "/" + filename) as f:
            data = json.load(f)
            # Extract just the world history part
            loadedHistory = data["worldHistory"]
            # Add this part to the world history
            worldHistory.extend(loadedHistory)

    return worldHistory


# Load all the JSON files from a given path
def loadDataZIPPED(path:str):
    filterPrefix = "discoveryworld-playlog"


    jsonFiles = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.zip')]
    # Filter out the files that don't start with the prefix
    jsonFiles = [file for file in jsonFiles if file.startswith(filterPrefix)]

    # The files will be of the form "<filename>.partXofY.json" -- Try to parse out the number of parts
    # and sort the files accordingly
    jsonFiles.sort()
    numParts = 0
    if len(jsonFiles) > 0:
        numParts = int(jsonFiles[-1].split(".part")[1].split("of")[1].split(".zip")[0])

    print("Number of parts: " + str(numParts))

    # Get the part of the filename before the ".partXofY.json" part
    filePrefix = jsonFiles[0].split(".part")[0]

    # Load each file, one by one
    worldHistory = []
    for i in range(numParts):
        filenameZIP = filePrefix + ".part" + str(i) + "of" + str(numParts) + ".zip"
        filenameJSON = filePrefix + ".part" + str(i) + "of" + str(numParts) + ".json"
        print("Loading file: " + filenameZIP)

        #with open(path + "/" + filename) as f:
        # Note: This is a zip file, so we need to extract the file f
        import zipfile
        with zipfile.ZipFile(path + "/" + filenameZIP, 'r') as zip_ref:
            zip_ref.extractall(path)
            # Keep track of the extracted files, to delete later
            print("Extracted files: " + str(zip_ref.namelist()))
            extractedFiles = zip_ref.namelist()

        with open(path + "/" + filenameJSON) as f:
            data = json.load(f)
            # Extract just the world history part
            loadedHistory = data["worldHistory"]
            # Add this part to the world history
            worldHistory.extend(loadedHistory)

        # Delete the extracted files
        for extractedFile in extractedFiles:
            print("Removing extracted file: " + extractedFile)
            os.remove(path + "/" + extractedFile)

    return worldHistory




def getAgents(step):
    grid = step["grid"]
    # Find the agent
    agents = []
    for row in grid:
        for cell in row:
            for obj in cell:
                if obj["name"] == "agent":
                    agents.append(obj)

    return agents

# High-level summary statistics
def summaryStatistics(worldHistory):
    statistics = {}

    # Get the last step
    lastStep = worldHistory[-1]

    # Total number of steps
    statistics["numSteps"] = len(worldHistory)

    # Number of actions the agent took
    agents = getAgents(lastStep)
    statistics["numAgents"] = len(agents)
    if (len(agents) > 0):
        # Get first agent
        agent = agents[0]

        # Number of actions the agent took
        agentActionHistory = agent["actionHistory"]
        statistics["actionHistory"] = agentActionHistory

    # Get whether or not the task was solved
    tasks = lastStep["taskScores"]
    task = tasks[0]
    taskName = task["taskName"]
    taskScoreNormalized = task["scoreNormalized"]
    taskCompleted = task["completed"]
    taskCompletedSuccessfully = task["completedSuccessfully"]

    statistics["taskName"] = taskName
    statistics["taskScoreNormalized"] = taskScoreNormalized
    statistics["taskCompleted"] = taskCompleted
    statistics["taskCompletedSuccessfully"] = taskCompletedSuccessfully

    # TODO: Total time to complete the task


    return statistics





#
#   Main
#
if __name__ == "__main__":
    #dataPath = "logs/discoveryworld-playlog-archaeology_dating_simple.seed0.20240423_152952"
    #dataPath = "logs/discoveryworld-playlog-archaeology_dating_simple.seed0.20240423_160045"
    dataPath = "logs/discoveryworld-playlog-archaeology_dating_simple.seed0.20240423_161942"
    #data = loadDataJSON(dataPath)   # Folder of JSON files
    data = loadDataZIPPED(dataPath) # Folder of ZIP files

    # Print the summary statistics
    print("Summary statistics:")
    summaryStats = summaryStatistics(data)
    print(json.dumps(summaryStats, indent=4))

    print("Done.")
