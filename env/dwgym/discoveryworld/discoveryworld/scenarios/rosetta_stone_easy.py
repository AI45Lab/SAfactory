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

# Test whether an agent can successfully complete a scenario that just involves traversing a dialog tree
def makeScenarioRosettaStoneEasy(world, numUserAgents=1):
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
    mkBuildingOneRoom(world, x=15, y=9, width=7, height=5, signText="PlanetX Store", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    tableLocations = [(16, 10), (17, 10), (18, 10), (19, 10), (20, 10)]                     # Back wall

    # Shuffle the table locations
    world.rng.shuffle(tableLocations)

    # Random task object names
    taskObjectNames = ["Mushroom", "Seed", "Microscope", "Shovel", "Dirt", "Spectrometer", "Thermometer", "RadiationMeter", "PetriDish", "Key", "QuantumCrystal", "FertilizerBag", "Coin", "MeasuringTape", "rock_glowing"]
    # Gibberish names: Blorpnik, Zinklef, Flubergast, Snarvle, Quibblit, Groblon, Wimblet, Grundle, Froznak, Splonker, Trankle, Jibbex, Plork, Snurfle, Vlimmox
    translationDict = {
        "Mushroom": "Blorpnik",
        "Seed": "Zinklef",
        "Microscope": "Flubergast",
        "Shovel": "Splonker",
        "Dirt": "Quibblit",
        "Spectrometer": "Wimblet",
        "Thermometer": "Grundle",
        "RadiationMeter": "Froznak",
        "PetriDish": "Dribblet",
        "Key": "Trankle",
        "QuantumCrystal": "Jibbex",
        "FertilizerBag": "Plork",
        "Coin": "Snurfle",
        "MeasuringTape": "Vlimmox",
        "rock_glowing": "Fribbet"
    }

    taskObjects = []
    for taskObjectName in taskObjectNames:
        taskObject = world.createObject(taskObjectName)
        taskObject.attributes["translation"] = translationDict[taskObjectName]      # Add the translation directly to the object attributes
        taskObjects.append(taskObject)

    # Shuffle the task objects
    world.rng.shuffle(taskObjects)

    # HACK: If the random seed is 3, shuffle again, since it gives similar results to another answer for this task
    if (world.randomSeed == 3):
        r = random.Random()
        r.seed(100)
        r.shuffle(taskObjects)

    # Make 5 tables along the wall
    for i in range(0, 5):
        # Get object name (translated)
        objectName = taskObjects[i].attributes["translation"]
        objectTable = world.createObject("TableWithSign", signText=objectName)
        #objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        # Add a random object to the table
        objectTable.addObject(taskObjects[i])
        if (i == 0):
            scoringInfo["objectToMove"] = taskObjects[i]


    # Add three agents
    NPCs = []
    # Names from Foundation
    agentNames = ["Gail", "Harry", "Yugo", "Dors", "Wanda"]
    agentName = agentNames[world.randomSeed % 5]
    agentLocations = [(16, 12), (17, 12), (19, 12), (20, 12)]
    # Shuffle
    world.rng.shuffle(agentLocations)

    # Add the agent
    npc = NPCGiveTest(world, agentName)
    world.addObject(agentLocations[0][0], agentLocations[0][1], Layer.AGENT, npc)
    world.addAgent(npc)
    mkDialogGiveTest(npc, objectToGive=scoringInfo["objectToMove"].attributes["translation"])
    scoringInfo["destinationContainer"] = npc
    NPCs.append(npc)

    # Critical hypothesis
    scoringInfo["criticalHypotheses"].append("The person (" + agentName + ") is using a word in another language (" + scoringInfo["objectToMove"].attributes["translation"] + ") to refer to the `" + scoringInfo["objectToMove"].name + "`")

    scoringInfo["criticalQuestions"].append("Does it clearly state the word `" + scoringInfo["objectToMove"].attributes["translation"] + "` refers to the `" + scoringInfo["objectToMove"].name + "`?")


    # Paths
    mkPathX(17, 14, 15, world)       # Town square to farm

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
        #world.addObject(18+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Put the agent in the next AgentLocation
        world.addObject(18, 12, Layer.AGENT, userAgent)
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("initial location", 18, 12)

    return scoringInfo



# NPC
class NPCGiveTest(NPC):
    def __init__(self, world, name):
        # Default sprite name
        randomSpritePrefixes = ["character32_", "character15_", "character16_", "character17_", "character35_", "character9_"]
        #self.spriteCharacterPrefix = random.choice(randomSpritePrefixes)
        randomPrefix = randomSpritePrefixes[world.randomSeed % 5]
        self.defaultSpriteName = randomPrefix + "agent_facing_east"
        print(f"NPCGiveTest: {self.defaultSpriteName}")

        super().__init__(world, name, defaultSpriteName=self.defaultSpriteName)

        self.spriteCharacterPrefix = randomPrefix


        # Rendering
        self.attributes["faceDirection"] = "east"
        #self.spriteCharacterPrefix = "character32_"
        # Randomly choose a sprite


        # Default attributes
        self.attributes["isMovable"] = False                       # Elder cannot be picked.
        self.attributes['isContainer'] = True                      # Elder's inventory is a container.
        self.attributes['isOpenable'] = False                      # Can't be opened
        self.attributes['isOpenContainer'] = False                 # Prevent other players from accessing its inventory.
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Dialog attributes
        self.attributes['isDialogable'] = True                    # Can it be dialoged with?

        # NPC States
        self.attributes['dialogAgentsSpokenWith'] = []             # List of dialog agents that this NPC has spoken with

        self.pathfinder = Pathfinder()

    def tick(self):
        if (self.tickCompleted):
            return

        # Debug
        #print(f"NPCElder (id: {self.name}): {self.attributes['states']}")
        super().tick()

        # Interpret any external states
        # No external states for this agent to interpret -- just waits for dialog



def mkDialogGiveTest(agent, objectToGive:str):
    tree = DialogTree(agent)

    rootNode = DialogNode("rootNode", "Hello! Could you please give me the `" + str(objectToGive) + "`?")
    rootNode.addDialogOption("I'll go get it for you.", "endDialog")
    tree.addNode(rootNode)
    tree.setRoot(rootNode.name)

    # Exit dialog
    endDialog = DialogNode("endDialog", "Thank you for talking with me.")
    tree.addNode(endDialog)

    # Store dialog tree in agent
    agent.setDialogTree(tree)


#
#   Task Scoring
#

class RosettaStoneTaskEasy(Task):

    def __init__(self, world, scoringInfo):
        self.objectToMove = scoringInfo["objectToMove"]
        self.destinationContainer = scoringInfo["destinationContainer"]

        #taskDescription = "Your task is to follow the instructions of the other person in the room.\n"
        taskDescription = "Your task is to talk to " + self.destinationContainer.name + " and follow their instructions.\n"

        super().__init__("RosettaStoneTaskEasy", taskDescription, world, scoringInfo)

        # Pick up the object
        self.scorecardPick = ScorecardElement("Pick up the object", "Pick up the " + self.objectToMove.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPick)

        # Give the object
        self.scorecardPlace = ScorecardElement("Give the object", "Give the " + self.objectToMove.name + " to " + self.destinationContainer.name + ".", maxScore=1)
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

        # Check to see if the object is in the destination container
        if (not self.scorecardPlace.completed):
            # Get the object's parent container
            parentContainer = self.objectToMove.parentContainer
            if (parentContainer != None):
                if (parentContainer.uuid == self.destinationContainer.uuid):
                    self.scorecardPlace.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid, self.destinationContainer.uuid], associatedNotes=f"Agent has placed the object {self.objectToMove.uuid} in the destination container (UUID: {self.destinationContainer.uuid}).")


        # Check for a failure condition: If there's anything inside the pot other than the object to move, the task is failed.
        failed = False
        destinationContainerContents = self.destinationContainer.contents
        if (self.scorecardPlace.completed == False) and (len(destinationContainerContents) > 0):
            failed = True


        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Set whether the task was completed or not (and whether it was completed successfully)
        if (failed):
            self.completed = True
            self.completedSuccessfully = False
        else:
            # Otherwise, blank these out
            self.completed = False
            self.completedSuccessfully = False

            # Check whether the task is complete
            # Here, the task is complete if all the objects have been collected and returned to the elder.
            if (self.score == self.maxScore):
                self.completed = True
                self.completedSuccessfully = True
