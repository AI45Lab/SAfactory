
import json
import os





# Main function
if __name__ == '__main__':

    # master path for data
    path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_easy/"
    #path = "/home/peter/github/discoveryworld/paper-results/react/output_dir_normal_partial/recoma/"
    #path = "/home/peter/github/discoveryworld/paper-results/react/challenge-from-marc/react_challenge/"

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
        seed = fields[2]
        key = taskName + "_" + difficulty + "_" + str(steps) + "steps"

        # Load the data
        with open(file, 'r') as file:
            data = json.load(file)

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

        out.append(packed)


    # Calculate averages across keys
    allKeys = set([x["key"] for x in out])
    averages = {}

    for key in allKeys:
        scores = [x["normalizedScore"] for x in out if x["key"] == key]
        completed = [x["completedSuccessfully"] for x in out if x["key"] == key]

        avgScore = sum(scores) / len(scores)
        avgCompleted = sum(completed) / len(completed)

        numSamples = len(scores)

        averages[key] = {
            "avgScore": avgScore,
            "avgCompleted": avgCompleted,
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