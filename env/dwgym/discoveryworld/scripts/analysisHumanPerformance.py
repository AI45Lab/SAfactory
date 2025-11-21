
import os
import json
import time
import zipfile
import ujson

# Get the scenario, difficulty, seed, and time from the filename
def extractInfoFromFilename(strIn):
    # format: discoveryworld-playlog-not_rocket_science_normal.seed0.20240527_174438.log.zip

    # remove everything before "discoveryworld-playlog-"
    strIn = strIn[strIn.find("discoveryworld-playlog-") + len("discoveryworld-playlog-"):]

    # remove everything after ".log.zip"
    strIn = strIn[:strIn.find(".log.zip")]

    # split by "."
    fields = strIn.split(".")
    scenarioAndDifficulty = fields[0]
    # "not_rocket_science_normal" -> "not rocket science" and "normal"
    # everything after last "_" is difficulty
    split1 = scenarioAndDifficulty.split("_")
    # last element is difficulty
    difficulty = split1[-1]
    # Difficulty must be 'easy', 'normal', or 'challenge'.
    # If it is not, then the difficulty is 'unknown'
    if difficulty not in ['easy', 'normal', 'challenge']:
        difficulty = 'unknown'
    # everything before last element is scenario
    scenario = " ".join(split1[:-1])
    if (difficulty == 'unknown'):
        scenario = scenarioAndDifficulty.replace("_", " ")

    seed = fields[1]
    # "seedX" -> X
    seed = int(seed[4:])

    dateAndTime = fields[2]
    # Convert to a time object
    timeObj = time.strptime(dateAndTime, "%Y%m%d_%H%M%S")

    return {
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "time": timeObj
    }


# Open a ZIP file, find all the JSON files in its directories (recursively), and load each one.
def loadPlaylogFromZip(zipPath):
    print("Processing ZIP file: " + zipPath)
    # Open the ZIP file
    worldHistoryOut = []

    # temporary directory
    tempDir = "tmp/"

    with zipfile.ZipFile(zipPath, 'r') as zip:
        # Find all the ZIP files within the ZIP file (that will be of the form ...part0of4, ...part1of4, etc.)
        partFiles = [f for f in zip.namelist() if f.endswith(".zip")]
        # Filter files that don't contain '.part'
        partFiles = [f for f in partFiles if ".part" in f]
        print("\t" + str(len(partFiles)) + " part ZIP files found")

        if (len(partFiles) == 0):
            print("### ERROR: No part files found in ZIP file: " + zipPath)
            return None

        # Find the maximum part number
        fileName1 = partFiles[0]
        # Remove ".zip" from the end
        fileName1 = fileName1[:-4]
        # Get everything after the last "."
        fileName1 = fileName1.split(".")[-1]
        # Get everything after the "of"
        fileName1 = fileName1.split("of")[-1]
        # Convert to an integer
        maxPart = int(fileName1)
        print("\tMax part number: " + str(maxPart))

        # Remove ".zip" from the end
        baseName = partFiles[0][:-4]
        # Remove ".partXofY" from the end
        baseName = baseName.split(".part")[0]


        # Load each part
        for i in range(maxPart):
            # Create a temporary directory to extract the ZIP files
            os.makedirs(tempDir, exist_ok=True)

            # Get part name
            partName = baseName + ".part" + str(i) + "of" + str(maxPart) + ".zip"
            print("\t\tProcessing part: " + partName)
            # Extract the part to the temporary directory
            zip.extract(partName, tempDir)
            print("\t\tExtracted part file: " + partName)

            # Load the ZIP file, and find all the JSON files in its directories (recursively)
            with zipfile.ZipFile(tempDir + partName, 'r') as partFile:
                print("\t\t\tOpened part file: " + partName)
                # Find all the JSON files in the ZIP file
                jsonFiles = [f for f in partFile.namelist() if f.endswith(".json")]
                print("\t\t\t" + str(len(jsonFiles)) + " JSON files found")
                # Load each JSON file
                for jsonFile in jsonFiles:
                    with partFile.open(jsonFile) as file:
                        print("\t\t\tFound JSON file: " + jsonFile)
                        #data = json.load(file)
                        data = ujson.load(file)
                        #print(data)
                        # extract 'worldHistory' key from data (should be a list)
                        # Resaving world history
                        worldHistory = data['worldHistory']
                        worldHistoryOut.extend(worldHistory)

                        #### Memory compression
                        removeAllNonAgentObjects(worldHistory)


            # Remove the temporary directory
            #os.system("rm -rf " + tempDir)
            # using python
            print("Removing temporary directory")
            #os.remove(tempDir)
            #os.system("rm -rf " + tempDir)
            # Remove every file that is in the directory
            for root, dirs, files in os.walk(tempDir):
                for file in files:
                    os.remove(os.path.join(root, file))

            for root, dirs, files in os.walk(tempDir):
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))

    print("Assembled world history with " + str(len(worldHistoryOut)) + " entries")
    return worldHistoryOut


# Try to compress the memory usage of the world history by removing all non-agent objects
def removeAllNonAgentObjects(worldHistory):
    print("Removing all non-agent objects from world history to conserve memory...")
    numRemoved = 0
    for idx, entry in enumerate(worldHistory):
        # Get the grid
        grid = entry['grid']
        # Find the agent
        for row in grid:
            for cell in row:
                for obj in cell:
                    if (obj["name"] != "agent"):
                        obj.clear()
                        numRemoved += 1

    print("Removed " + str(numRemoved) + " non-agent objects")


def findCompletedIndexInWorldHistory(worldHistory):
    for idx, entry in enumerate(worldHistory):
        if ("taskScores" not in entry):
            continue
        if (len(entry["taskScores"]) == 0):
            continue
        if (entry["taskScores"][0]["completed"]):
            return idx
    return -1

# Find agent in world history
def findAgentInWorldHistory(worldHistory, idx=-1):
    # Get last world history entry
    lastEntry = worldHistory[idx]
    # Get the grid
    grid = lastEntry['grid']
    # Find the agent
    for row in grid:
        for cell in row:
            for obj in cell:
                if ("name" in obj) and (obj["name"] == "agent"):
                    return obj
    return None

def findTaskScoresInWorldHistory(worldHistory, idx=-1):
    # Get last world history entry
    lastEntry = worldHistory[-1]
    # Get the grid
    taskScores = lastEntry['taskScores']
    return taskScores[0]



# Main
if __name__ == '__main__':
    # Base path for data
    #pathIn = "/home/peter/Documents/discoveryworld-userstudy-data/may28-2024-partial/"
    pathIn = "/home/peter/Documents/discoveryworld-userstudy-data/june3-2024-full/"

    # Participants (in subdirectories)
    subjects = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"]

    out = []

    startTime = time.time()

    # For each participant, load data
    for subject in subjects:
        # Load data
        dataPath = pathIn + subject + "/"
        # Crawl the directory recursively to find all files
        files = []
        for root, dirs, fileNames in os.walk(dataPath):
            for fileName in fileNames:
                files.append(root + "/" + fileName)

        # Filter files to those that contain "discoveryworld-playlog" and end with ".zip"
        files = [f for f in files if "discoveryworld-playlog" in f and f.endswith(".zip")]
        # Do not include any files that contain "part" in the name
        files = [f for f in files if ".part" not in f]

        print("Subject: " + subject)
        print("\t" + str(len(files)) + " files found")
        for f in files:
            deltaTime = time.time() - startTime
            print("\t" + str(deltaTime) + " seconds elapsed")
            print("\t" + f)
            #print("\t" + str(extractInfoFromFilename(f)))
            information = extractInfoFromFilename(f)
            print("\tScenario: " + information["scenario"])
            print("\tDifficulty: " + information["difficulty"])
            print("\tSeed: " + str(information["seed"]))
            print("\tTime: " + time.strftime("%Y-%m-%d %H:%M:%S", information["time"]))

            information["file"] = f
            information["participant"] = subject
            information["errors"] = []

            # Load the playlog from the ZIP file
            worldHistory = None
            try:
                worldHistory = loadPlaylogFromZip(f)
            # Keyboard exception
            except KeyboardInterrupt:
                print("### Keyboard interrupt")
                exit(1)
            except Exception as e:
                print("### ERROR: Could not load playlog from ZIP file")
                print(e)
                information["errors"].append("Error loading playlog from ZIP file")

            if (worldHistory is not None):
                # Find the step that the task was completed on (if it was)
                completedIndex = findCompletedIndexInWorldHistory(worldHistory)
                information["completedIndex"] = completedIndex
                print("\tTask was completed on: " + str(completedIndex))

                agent = findAgentInWorldHistory(worldHistory, completedIndex)
                if (agent is None):
                    print("### ERROR: Agent not found in world history")
                    continue

                actionHistory = agent['actionHistory']
                print("\tAgent action history: " + str(len(actionHistory)) + " actions")

                # Calculate statistics
                stats = {
                    "totalActions": len(actionHistory),
                    "moveActions": 0,
                    "nonMoveActions": 0,
                }
                for idx, action in enumerate(actionHistory):
                    #print("\t\t" + action['action'] + ": " + str(action['success']))
                    print("\t" + str(action))
                    if (action["actionType"].startswith("MOVE") or action["actionType"].startswith("ROTATE") or action["actionType"].startswith("TELEPORT")):
                        stats["moveActions"] += 1
                    else:
                        stats["nonMoveActions"] += 1

                information["stats"] = stats

                # Find task scores
                taskScores = findTaskScoresInWorldHistory(worldHistory, completedIndex)
                print("\tTask scores: " + str(taskScores))
                information["taskScores"] = taskScores
                information["rawActionHistory"] = actionHistory

            else:
                information["errors"].append("Error processing this file.")

            out.append(information)

            # Write to file
            filenameOut = pathIn + "/analysisHumanPerformance.json"
            with open(filenameOut, 'w') as fileOut:
                json.dump(out, fileOut, indent=4)

            #exit(1)