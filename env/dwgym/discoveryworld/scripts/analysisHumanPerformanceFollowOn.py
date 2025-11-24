

import os
import time
import json





# Main
if __name__ == '__main__':

    # Path to the summarized human performance
    #path = "/home/peter/github/discoveryworld/paper-results/human/"
    #dataFilename = "analysisHumanPerformance.june3full.json"
    path = "/home/peter/github/discoveryworld/paper-results/human/corrected-rocket-science/"
    dataFilename = "analysisHumanPerformance.june3full.new2.json"


    # Load the data
    print("Loading data from: " + path + dataFilename)
    with open(path + dataFilename, 'r') as file1:
        data = json.load(file1)


    numRecords = len(data)
    print("Found " + str(numRecords) + " records")

    # Data by participant
    dataByParticipant = {}
    for record in data:
        participant = record["participant"]
        if (participant not in dataByParticipant):
            dataByParticipant[participant] = []
        dataByParticipant[participant].append(record)

    # Summary
    participants = dataByParticipant.keys()
    for key in participants:
        print("Participant: " + key + " (" + str(len(dataByParticipant[key])) + " records)")


    # For each participant, keep only the first record for each "scenario" and "difficulty" combination.  There's a datestamp in the `time` key.
    # The datestamp was saved as an array using the following code: timeObj = time.strptime(dateAndTime, "%Y%m%d_%H%M%S")
    # Example of saved data:
    # "time": [
    #     2024,
    #     5,
    #     2,
    #     1,
    #     47,
    #     20,
    #     3,
    #     123,
    #     -1
    # ],
    dataBinned = {}
    out = []
    allKeys = set()
    for participant in participants:
        dataBinned[participant] = {}
        knownKeys = set()

        for record in dataByParticipant[participant]:
            # Convert the time list to a tuple before using time.mktime()
            record["time"] = time.mktime(tuple(record["time"]))

        # Sort into bins by scenario and difficulty
        for record in dataByParticipant[participant]:
            key = record["scenario"] + "_" + record["difficulty"]

            # Some keys changed during development -- renormalize them.
            key = key.replace("combinatorial chemistry_unknown", "combinatorial chemistry_normal")
            key = key.replace("lost in translation_easy", "lost in translation_normal")
            key = key.replace("food illness_unknown", "space sick_normal")

            knownKeys.add(key)
            allKeys.add(key)
            if key not in dataBinned[participant]:
                dataBinned[participant][key] = record
            else:
                if record["time"] < dataBinned[participant][key]["time"]:
                    dataBinned[participant][key] = record

            scores = {
                "error": "error processing this record"
            }

            try:
                # Analysis summary -- only keep things we might report in the paper
                completedSuccessfully = record["taskScores"]["completedSuccessfully"]
                if (completedSuccessfully == True):
                    completedSuccessfully = 1
                else:
                    completedSuccessfully = 0

                totalActions = record["stats"]["totalActions"]
                moveActions = record["stats"]["moveActions"]
                nonMoveActions = record["stats"]["nonMoveActions"]
                propMoveActions = 0
                if (totalActions > 0):
                    propMoveActions = moveActions / totalActions
                propNonMoveActions = 0
                if (totalActions > 0):
                    propNonMoveActions = nonMoveActions / totalActions

                scores = {
                    "scoreNormalized": record["taskScores"]["scoreNormalized"],
                    "completedSuccessfully": completedSuccessfully,
                    "totalActions": totalActions,
                    "moveActions": moveActions,
                    "nonMoveActions": nonMoveActions,
                    "propMoveActions": propMoveActions,
                    "propNonMoveActions": propNonMoveActions,
                }

            except:
                pass

            dataBinned[participant][key]["summary_statistics"] = scores


        # Summary
        print("")
        print("-" * 80)
        print("")
        print("Participant: " + participant)
        knownKeys = sorted(list(knownKeys))
        print("Known Keys: " + str(knownKeys))
        print("")
        for key in knownKeys:
            print("Key: " + key)
            #print(dataBinned[participant][key])
            print("Summary: " + str(dataBinned[participant][key]["summary_statistics"]))
            import copy
            packed = copy.deepcopy(dataBinned[participant][key]["summary_statistics"])
            packed["key"] = key
            packed["participant"] = participant
            out.append(packed)
            print("")


    # Export the data as a TSV
    #filenameOut = path + "analysisHumanPerformanceFollowOn.tsv"
    filenameOut = path + dataFilename.replace(".json", "FollowOn.tsv")
    print("Exporting data to: " + filenameOut)

    # Sort data by key, then secondarily by participant
    out = sorted(out, key=lambda x: (x["key"], x["participant"]))

    with open(filenameOut, 'w') as file1:
        keys = out[0].keys()
        file1.write("\t".join(keys) + "\n")
        for record in out:
            row = []
            for key in keys:
                if (key in record):
                    row.append(str(record[key]))
                else:
                    row.append("")
            file1.write("\t".join(row) + "\n")


    # Filter out any data with missing scores
    out = [x for x in out if "scoreNormalized" in x]

    # Calculate averages within keys
    averages = {}
    for key in allKeys:
        # Get all data for this key across participants
        scores = [x["scoreNormalized"] for x in out if x["key"] == key]
        completed = [x["completedSuccessfully"] for x in out if x["key"] == key]
        totalActions = [x["totalActions"] for x in out if x["key"] == key]
        moveActions = [x["moveActions"] for x in out if x["key"] == key]
        nonMoveActions = [x["nonMoveActions"] for x in out if x["key"] == key]
        propMoveActions = [x["propMoveActions"] for x in out if x["key"] == key]
        propNonMoveActions = [x["propNonMoveActions"] for x in out if x["key"] == key]

        avgScore = sum(scores) / len(scores)
        avgCompleted = sum(completed) / len(completed)
        avgTotalActions = sum(totalActions) / len(totalActions)
        avgMoveActions = sum(moveActions) / len(moveActions)
        avgNonMoveActions = sum(nonMoveActions) / len(nonMoveActions)
        avgPropMoveActions = sum(propMoveActions) / len(propMoveActions)
        avgPropNonMoveActions = sum(propNonMoveActions) / len(propNonMoveActions)

        numSamples = len(scores)
        averages[key] = {
            "avgScore": avgScore,
            "avgCompleted": avgCompleted,
            "avgTotalActions": avgTotalActions,
            "avgMoveActions": avgMoveActions,
            "avgNonMoveActions": avgNonMoveActions,
            "avgPropMoveActions": avgPropMoveActions,
            "avgPropNonMoveActions": avgPropNonMoveActions,
            "numSamples": numSamples
        }



    # Export the averages, too
    #filenameOut = path + "analysisHumanPerformanceFollowOnAverages.tsv"
    filenameOut = path + dataFilename.replace(".json", ".tsv")
    print("Exporting averages to: " + filenameOut)
    with open(filenameOut, 'w') as file1:
        keys = averages[list(averages.keys())[0]].keys()
        file1.write("key\t" + "\t".join(keys) + "\n")
        for key in sorted(averages.keys()):
            row = [key]
            for key2 in keys:
                row.append(str(averages[key][key2]))
            file1.write("\t".join(row) + "\n")