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

from discoveryworld.ActionSuccess import ActionSuccess, MessageImportance

# Test whether an agent can successfully complete a scenario that just involves traversing a dialog tree
def makeScenarioSpaceSickEasy(world, numUserAgents=1):
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
    mkBuildingOneRoom(world, x=15, y=10, width=7, height=5, signText="Ministry of Measurement", includeDoor=True, doorKeyID = DOOR_KEY_ID)


    # Along the top wall, make a row of tables with 5 measuring devices on them
    #measurementDeviceNames = ["Densitometer", "Spectrometer", "Thermometer", "Microscope", "PHMeter"]
            # "SimplifiedMicroscope": SimplifiedScientificInstrument["microscope"],
            # "SimplifiedSpectrometer": SimplifiedScientificInstrument["spectrometer"],
            # "SimplifiedPHMeter": SimplifiedScientificInstrument["ph_meter"],
            # "SimplifiedRadiationMeter": SimplifiedScientificInstrument["radiation meter"],
            # "SimplifiedDensitometer": SimplifiedScientificInstrument["densitometer"],
    measurementDeviceNames = ["SimplifiedDensitometer", "SimplifiedSpectrometer", "SimplifiedPHMeter", "SimplifiedMicroscope", "SimplifiedRadiationMeter"]
    measurementDevicesDict = {}
    measurementDevices = []
    for i, deviceName in enumerate(measurementDeviceNames):
        device = world.createObject(deviceName)
        measurementDevicesDict[deviceName] = device
        measurementDevices.append(device)
        if (i == (world.randomSeed % 5)):
            device.attributes["isCriticalInstrument"] = True
            scoringInfo["criticalInstrument"] = device

    scoringInfo["measurementDevices"] = measurementDevices

    world.rng.shuffle(measurementDevices)
    tableLocations = [(16, 11), (17, 11), (18, 11), (19, 11), (20, 11)]
    world.rng.shuffle(tableLocations)
    for i in range(0, 5):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        # Add a random object to the table
        objectTable.addObject(measurementDevices[i])


    possiblePathogens = ["Bacterium Species 138", "Virus Strain 12", "Fungus Species 3", "Parasite Species 7", "Prion Strain 2"]
    currentPathogen = possiblePathogens[world.randomSeed%5]

    scoringInfo["criticalHypotheses"].append("One of the food items is contaminated with `" + currentPathogen + "`, which can be measured with the " + scoringInfo["criticalInstrument"].name + ".")
    scoringInfo["criticalQuestions"].append("Does it clearly state that one of the food items is contaminated with a pathogen (`" + currentPathogen + "`)?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that the pathogen can be measured/detected with the " + scoringInfo["criticalInstrument"].name + "?")

    # Random task object names
    taskObjects = []
    for i in range(0, 3):
        taskObject = world.createObject("Mushroom")
        if (i == 0):
            taskObject.attributes["pathogen"] = currentPathogen
            scoringInfo["objectToMove"] = taskObject
        taskObjects.append(taskObject)
    #world.rng.shuffle(taskObjects)
    scoringInfo["taskObjects"] = taskObjects

    tableLocations = [(16, 13), (17, 13), (19, 13), (20, 13)]
    for i in range(0, (world.randomSeed%3)+1):
        world.rng.shuffle(tableLocations)
    for i in range(0, 3):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        # Add a random object to the table
        objectTable.addObject(taskObjects[i])


    destinationContainer = world.createObject("Jar")
    containerTable = world.createObject("Table")
    containerTable.addObject(destinationContainer)
    world.addObject(tableLocations[3][0], tableLocations[3][1], Layer.OBJECTS, containerTable)
    scoringInfo["destinationContainer"] = destinationContainer


        # TODO: This part isn't correct
        #if (i == 0):
        #    # Make the first randomly placed object the object to move
        #    scoringInfo["destinationContainer"] = destinationContainers[i]

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

class SpaceSickTaskEasy(Task):

    def __init__(self, world, scoringInfo):
        self.objectToMove = scoringInfo["objectToMove"]
        self.destinationContainer = scoringInfo["destinationContainer"]
        self.scoringInfo = scoringInfo

        taskDescription = "Some of the food on Planet X has been contaminated with a pathogen that is causing mild stomach illness. "
        taskDescription += "You must determine which of the 3 mushrooms is contaminated, then place the contaminated mushroom in the jar for further analysis and disposal."

        super().__init__("SmallSkillsInstrumentMeasurementTask", taskDescription, world, scoringInfo)

        # Scorecard elements
        # Pick
        self.scorecardPick = ScorecardElement("Pick up infected food", "Pick up the " + self.objectToMove.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPick)

        # Measure
        self.scorecardMeasure = ScorecardElement("Use measurement devices", "Use each measurement device with at least one food item.", maxScore=5)
        self.scoreCard.append(self.scorecardMeasure)

        self.scorecardMeasureTaskCritical = ScorecardElement("Measure pathogen", "Use the critical measurement device on the contaminated food item.", maxScore=1)
        self.scoreCard.append(self.scorecardMeasureTaskCritical)

        # Place
        self.scorecardPlace = ScorecardElement("Place the object", "Place the infected food in the " + self.destinationContainer.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPlace)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

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
            #instrument = self.scoringInfo["measurementDevice"]
            usedDevices = set()
            for instrument in self.scoringInfo["measurementDevices"]:
                for agent in self.world.getUserAgents():
                    for taskObject in self.scoringInfo["taskObjects"]:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2=taskObject, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            usedDevices.add(instrument.name)

            if (len(usedDevices) == 5):
                self.scorecardMeasure.updateScore(5, True, associatedUUIDs=[self.objectToMove.uuid], associatedNotes=f"Agent has used all instruments at least once.")
            else:
                self.scorecardMeasure.updateScore(len(usedDevices), False, associatedUUIDs=[self.objectToMove.uuid], associatedNotes=f"Agent has used the following instruments: {str(usedDevices)}.")

        # Check to see if the object has been measured by the correct instrument
        if (not self.scorecardMeasureTaskCritical.completed):
            for agent in self.world.getUserAgents():
                foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["criticalInstrument"], arg2=self.objectToMove, stopAtFirst=True)
                if (len(foundActions) > 0):
                    self.scorecardMeasureTaskCritical.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid], associatedNotes=f"Agent has used the critical instrument on the contaminated food item.")

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

        if (failure):
            self.completed = True
            self.completedSuccessfully = False
        else:
            # Otherwise, blank these out
            self.completed = False
            self.completedSuccessfully = False

            # Check whether the task is complete
            if (self.scorecardPlace.completed == True):
                self.completed = True
                self.completedSuccessfully = True

        # Update score
        self.score = sum(element.score for element in self.scoreCard)



#
#   Simplified Scientific Instrument
#

class SimplifiedScientificInstrument(Object):
    # Constructor
    def __init__(self, world, instrumentType:str):
        # Use the same class to represent all simplified instruments
        simpifiedInstruments = {
            "densitometer": {
                "name": "densitometer",
                "type": "densitometer",
                "defaultSpriteName": "instruments_densitometer"
            },
            "microscope": {
                "name": "microscope",
                "type": "microscope",
                "defaultSpriteName": "instruments_microscope"
            },
            "ph_meter": {
                "name": "PH meter",
                "type": "PH meter",
                "defaultSpriteName": "instruments_ph_meter"
            },
            "spectrometer": {
                "name": "spectrometer",
                "type": "spectrometer",
                "defaultSpriteName": "instruments_spectrometer"
            },
            "radiation_meter": {
                "name": "radiation meter",
                "type": "radiation meter",
                "defaultSpriteName": "instruments_radiation_meter"
            }
        }

        if (instrumentType not in simpifiedInstruments):
            print("ERROR: Invalid instrument type: " + instrumentType)
            exit(1)
        name = simpifiedInstruments[instrumentType]["name"]
        type = simpifiedInstruments[instrumentType]["type"]
        defaultSpriteName = simpifiedInstruments[instrumentType]["defaultSpriteName"]

        Object.__init__(self, world, objectType=type, objectName=name, defaultSpriteName=defaultSpriteName)

        # Default attributes

        self.attributes['isUsable'] = True                       # Can this device be used with another object? (e.g. specifically through the 'use' action)

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Whether or not this instrument is the "critical instrument" that can detect pathogens or not
        self.attributes["isCriticalInstrument"] = False


    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        outputDescriptionStr = "You use the " + self.name + " to observe the " + patientObj.name + ".\n"

        # Check to see if the patient object is infected with any pathogen
        if ("pathogen" in patientObj.attributes) and (self.attributes["isCriticalInstrument"] == True):
            pathogenName = patientObj.attributes["pathogen"]
            outputDescriptionStr += "The readings appear consistent with a " + patientObj.name + " that has been contaminated with " + pathogenName + ".\n"
        else:
            outputDescriptionStr += "The readings appear normal.\n"

        # Return the action response
        return ActionSuccess(True, outputDescriptionStr, importance=MessageImportance.HIGH)

    #
    #   Tick
    #
    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName
