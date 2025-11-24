import random

from discoveryworld.Agent import NPC, Agent
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer
from discoveryworld.Pathfinding import Pathfinder
from discoveryworld.buildings.colony import mkStorageShed, mkStorageShedChallenge
from discoveryworld.buildings.terrain import mkGrassFill, mkPathX, mkTallTree
from discoveryworld.buildings.house import mkBuildingDivided, mkBuildingOneRoom, mkTableAndChairs
from discoveryworld.DialogTree import DialogTree, DialogNode
from discoveryworld.TaskScorer import Task, ScorecardElement
from discoveryworld.ActionHistory import *

# Test whether an agent can successfully complete a scenario that just involves traversing a dialog tree
def makeScenarioInstrumentMeasurementTest(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    DOOR_KEY_ID = 123

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 1
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects
    # Fill with grass
    mkGrassFill(world)
    # Randomly place a few plants (plant1, plant2, plant3)
    for i in range(0, 10):
        randY = world.rng.randint(0, world.sizeY - 1)
        randX = world.rng.randint(0, world.sizeX - 1)

    # Buildings
    mkBuildingOneRoom(world, x=14, y=9, width=9, height=6, signText="Ministry of Measurement", includeDoor=True, doorKeyID = DOOR_KEY_ID)


    # Along the top wall, make a row of tables with 5 measuring devices on them
    measurementDeviceNames = ["Densitometer", "Spectrometer", "Thermometer", "Microscope", "PHMeter"]
    measurementDevicesDict = {}
    measurementDevices = []
    for deviceName in measurementDeviceNames:
        device = world.createObject(deviceName)
        measurementDevicesDict[deviceName] = device
        measurementDevices.append(device)

    world.rng.shuffle(measurementDevices)
    tableLocations = [(16, 10), (17, 10), (18, 10), (19, 10), (20, 10)]
    world.rng.shuffle(tableLocations)
    for i in range(0, 5):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        # Add a random object to the table
        objectTable.addObject(measurementDevices[i])


    # Random task object names
    taskObjectNames = ["Mushroom", "Seed", "Shovel", "Dirt", "Key", "FertilizerBag", "rock_glowing"]
    taskObjects = [world.createObject(taskObjectName) for taskObjectName in taskObjectNames]
    world.rng.shuffle(taskObjects)
    tableLocations = [(16, 13), (17, 13), (19, 13), (20, 13)]
    world.rng.shuffle(tableLocations)

    microscopeDescriptions = ["scratched", "rough", "smooth", "bumpy"]
    for i in range(0, 4):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        # Add a random object to the table
        objectTable.addObject(taskObjects[i])
        if (i == 0):
            # Make the first randomly placed object the object to move
            scoringInfo["objectToMove"] = taskObjects[i]

        # Randomize the material properties of the task objects
        randomMaterial = {}
        taskObjects[i].attributes["density"] = round(world.rng.uniform(1, 10.0), 1)
        randomMaterial["ph"] = round(world.rng.uniform(1, 10.0), 1)
        taskObjects[i].attributes["temperatureC"] = round(world.rng.uniform(1, 10.0), 1)
        randomMaterial["spectrum"] = []
        for j in range(0, 5):
            randomMaterial["spectrum"].append(round(world.rng.uniform(1, 10.0), 1))
        randomMicroscopeDesc = world.rng.choice(microscopeDescriptions)
        randomMaterial["microscopeDesc"] = "a " + str(randomMicroscopeDesc) + " surface"
        microscopeDescriptions.remove(randomMicroscopeDesc)

        # Remove any existing material properties, replace with this one
        taskObjects[i].attributes['materials'] = [randomMaterial]

        #print("Task object: " + taskObjects[i].name)
        #print("Material properties: " + str(taskObjects[i].attributes['materials']))


    # Destination containers
    #destinationContainerNames = ["Pot", "Jar"]
    destinationContainers = []
    for i in range(0, 4):
         container = world.createObject("Pot")
         # make pot A, b, c, d, etc.
         container.name = "pot " + chr(65 + i)
         destinationContainers.append(container)
    tableLocations = [(15, 11), (15, 12), (21, 12), (21, 11)]

    for i in range(0, 4):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)
        objectTable.addObject(destinationContainers[i])

        # TODO: This part isn't correct
        #if (i == 0):
        #    # Make the first randomly placed object the object to move
        #    scoringInfo["destinationContainer"] = destinationContainers[i]



    # Randomly pick a target property
    randomProperty = world.rng.choice(["density", "ph", "temperatureC", "spectrum", "microscopeDesc"])
    propertyToMeasureStr = ""
    targetValue = None
    if (randomProperty == "density"):
        targetValue = scoringInfo["objectToMove"].attributes["density"]
        propertyToMeasureStr = "density"
        scoringInfo["measurementDevice"] = measurementDevicesDict["Densitometer"]
    elif (randomProperty == "ph"):
        targetValue = scoringInfo["objectToMove"].attributes["materials"][0]["ph"]
        propertyToMeasureStr = "PH"
        scoringInfo["measurementDevice"] = measurementDevicesDict["PHMeter"]
    elif (randomProperty == "temperatureC"):
        targetValue = scoringInfo["objectToMove"].attributes["temperatureC"]
        propertyToMeasureStr = "temperature"
        scoringInfo["measurementDevice"] = measurementDevicesDict["Thermometer"]
    elif (randomProperty == "spectrum"):
        targetValue = scoringInfo["objectToMove"].attributes["materials"][0]["spectrum"][2]
        propertyToMeasureStr = "spectrum (on channel 3)"
        scoringInfo["measurementDevice"] = measurementDevicesDict["Spectrometer"]
    elif (randomProperty == "microscopeDesc"):
        targetValue = scoringInfo["objectToMove"].attributes["materials"][0]["microscopeDesc"]
        propertyToMeasureStr = "microscopic description"
        scoringInfo["measurementDevice"] = measurementDevicesDict["Microscope"]
    else:
        print("ERROR: This condition should never be met.")
        exit(1)

    scoringInfo["taskText"] = "Your task is measure the " + propertyToMeasureStr + " of the `" + scoringInfo["objectToMove"].name + "`, and place it in a specific container depending on the measured value.\n"
    # Target container
    if (type(targetValue) == float):
        if (targetValue <= 2.5):
            targetContainer = destinationContainers[0]
        elif (targetValue <= 5.0):
            targetContainer = destinationContainers[1]
        elif (targetValue <= 7.5):
            targetContainer = destinationContainers[2]
        else:
            targetContainer = destinationContainers[3]

        scoringInfo["taskText"] += "If the value is less than 2.5, place it in pot A.\n"
        scoringInfo["taskText"] += "If the value is between 2.5 and 5.0, place it in pot B.\n"
        scoringInfo["taskText"] += "If the value is between 5.0 and 7.5, place it in pot C.\n"
        scoringInfo["taskText"] += "If the value is greater than 7.5, place it in pot D.\n"

    else:
        if (targetValue == "a rough surface"):
            targetContainer = destinationContainers[0]
        elif (targetValue == "a smooth surface"):
            targetContainer = destinationContainers[1]
        elif (targetValue == "a bumpy surface"):
            targetContainer = destinationContainers[2]
        else:
            targetContainer = destinationContainers[3]

        scoringInfo["taskText"] += "If the object has a rough surface, place it in pot A.\n"
        scoringInfo["taskText"] += "If the object has a smooth surface, place it in pot B.\n"
        scoringInfo["taskText"] += "If the object has a bumpy surface, place it in pot C.\n"
        scoringInfo["taskText"] += "If the object has any other kind of surface, place it in pot D.\n"

    scoringInfo["destinationContainer"] = targetContainer
    scoringInfo["distractorContainers"] = [container for container in destinationContainers if container != targetContainer]



    # Paths
    mkPathX(17, 15, 15, world)       # Town square to farm

    mkTallTree(13, 10, world)
    mkTallTree(21, 8, world)
    mkTallTree(17, 6, world)
    mkTallTree(15, 15, world)
    mkTallTree(18, 20, world)
    mkTallTree(22, 11, world)
    mkTallTree(26, 16, world)
    mkTallTree(29, 16, world)
    mkTallTree(13, 18, world)
    mkTallTree(14, 16, world)
    mkTallTree(10, 9, world)
    mkTallTree(8, 11, world)
    mkTallTree(7, 15, world)
    mkTallTree(28, 8, world)

    # Add some plants
    world.addObject(15, 1, Layer.OBJECTS, world.createObject("PlantGeneric"))
    plantCount = 0
    minPlants = 15
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a plant
                world.addObject(randX, randY, Layer.OBJECTS, world.createObject("PlantGeneric"))
                plantCount += 1


    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # TODO: ADD KEY

        # Add the agent to a specfic location
        world.addObject(18+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("initial location", 18, 12)


    return scoringInfo



#
#   Task Scoring
#

class SmallSkillsInstrumentMeasurementTask(Task):

    def __init__(self, world, scoringInfo):
        self.objectToMove = scoringInfo["objectToMove"]
        self.destinationContainer = scoringInfo["destinationContainer"]
        self.scoringInfo = scoringInfo

        taskDescription = scoringInfo["taskText"]

        super().__init__("SmallSkillsInstrumentMeasurementTask", taskDescription, world, scoringInfo)

        # Scorecard elements
        # Pick
        self.scorecardPick = ScorecardElement("Pick up the object", "Pick up the " + self.objectToMove.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPick)

        # Measure
        self.scorecardMeasure = ScorecardElement("Measure the object", "Measure the " + self.objectToMove.name + " with the " + scoringInfo["measurementDevice"].name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardMeasure)

        # Place
        self.scorecardPlace = ScorecardElement("Place the object", "Place the " + self.objectToMove.name + " in " + self.destinationContainer.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPlace)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        pass

    def updateTick(self):
        if self.completed:
            return  # Do not update the score if the task is already marked as completed

        # Check to see if the object is in the agent's inventory
        if (not self.scorecardPick.completed):
            # Get the object's parent container
            parentContainer = self.objectToMove.parentContainer
            if (parentContainer != None):
                if (parentContainer.attributes["isAgent"] == True):
                    self.scorecardPick.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid], associatedNotes=f"Agent has picked up the object (UUID: {self.objectToMove.uuid}).")

        # Check to see if the object has been measured by the correct instrument
        if (not self.scorecardMeasure.completed):
            instrument = self.scoringInfo["measurementDevice"]
            for agent in self.world.getUserAgents():
                foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2=self.objectToMove, stopAtFirst=True)
                if (len(foundActions) > 0):
                    self.scorecardMeasure.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid, instrument.uuid], associatedNotes=f"Agent has measured the object {self.objectToMove.uuid} with the correct instrument (UUID: {instrument.uuid}).")

        # Check to see if the object is in the destination container
        if (not self.scorecardPlace.completed):
            # Get the object's parent container
            parentContainer = self.objectToMove.parentContainer
            if (parentContainer != None):
                if (parentContainer.uuid == self.destinationContainer.uuid):
                    self.scorecardPlace.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid, self.destinationContainer.uuid], associatedNotes=f"Agent has placed the object {self.objectToMove.uuid} in the destination container (UUID: {self.destinationContainer.uuid}).")


        failure = False
        # Check for a failure condition: If there's anything inside the pot other than the object to move, the task is failed.
        destinationContainerContents = self.destinationContainer.contents
        if (self.scorecardPlace.completed == False) and (len(destinationContainerContents) > 0):
            failure = True

        # Check for failure condition: Any object is in the distractor containers
        for container in self.scoringInfo["distractorContainers"]:
            if (len(container.contents) > 0):
                failure = True


        if (failure):
            self.completed = True
            self.completedSuccessfully = False
        else:
            # Otherwise, blank these out
            self.completed = False
            self.completedSuccessfully = False

        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        if (self.score == self.maxScore):
            self.completed = True
            self.completedSuccessfully = True
