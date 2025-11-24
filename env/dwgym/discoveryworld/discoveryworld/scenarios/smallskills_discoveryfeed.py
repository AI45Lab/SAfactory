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
def makeScenarioDiscoveryFeedTest(world, numUserAgents=1):
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
    mkBuildingOneRoom(world, x=15, y=9, width=7, height=6, signText="Discovery Feed Station", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    tableLocations = [(17, 10), (18, 10), (19, 10)]                     # Back wall
    tableLocations.extend([(16, 11), (16, 12), (20, 11), (20, 12)])     # Side walls
    tableLocations.extend([(17, 13), (19, 13)])                         # Front walls

    # Shuffle the table locations
    world.rng.shuffle(tableLocations)


    # Random task object names
    taskObjectNames = ["Mushroom", "Seed", "Microscope", "Shovel", "Dirt", "FlowerPot", "Spectrometer", "Thermometer", "RadiationMeter", "PetriDish", "Key", "QuantumCrystal", "FertilizerBag", "Coin", "MeasuringTape", "rock_glowing"]
    taskObjects = []
    for taskObjectName in taskObjectNames:
        taskObjects.append(world.createObject(taskObjectName))

    # Shuffle the task objects
    world.rng.shuffle(taskObjects)

    # Destination containers
    destinationContainerNames = ["Pot", "Jar"]
    destinationContainers = []
    for destinationContainerName in destinationContainerNames:
        destinationContainers.append(world.createObject(destinationContainerName))

    # Shuffle the destination containers
    world.rng.shuffle(destinationContainers)

    # Make 6 tables along the wall
    distractorObjects = []
    for i in range(0, 6):
        objectTable = world.createObject("Table")
        world.addObject(tableLocations[i][0], tableLocations[i][1], Layer.OBJECTS, objectTable)

        if (i < 5):
            # Add a random object to the table
            objectTable.addObject(taskObjects[i])
            if (i == 0):
                # Make the first randomly placed object the object to move
                scoringInfo["objectToMove"] = taskObjects[i]
            else:
                distractorObjects.append(taskObjects[i])
        else:
            # Add a random destination container to the table
            objectTable.addObject(destinationContainers[0])
            scoringInfo["destinationContainer"] = destinationContainers[0]

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

    # Discovery feed -- add (1) a post to pay attention to, (2) distractor posts
    # Agent names
    discoveryFeedNames = ["Hari", "Bayta", "Salvor", "Arkady", "Hober", "Wanda", "Gaal", "Dors"]
    # Shuffle
    world.rng.shuffle(discoveryFeedNames)
    # Correct post
    correctPost = {"name": discoveryFeedNames[0], "post": "Please pick up the " + scoringInfo["objectToMove"].name + " and place it in the " + scoringInfo["destinationContainer"].name + "."}
    scoringInfo["correctPostName"] = discoveryFeedNames[0]
    # Distactor posts
    distractorPosts = []
    for nameIdx, distractor in enumerate(distractorObjects):
        distractorPosts.append({"name": discoveryFeedNames[nameIdx+1], "post": "Please pick up the " + distractor.name + " and place it in the " + destinationContainers[0].name + "."})
    # Add posts to the discovery feed
    allPosts = [correctPost] + distractorPosts
    world.rng.shuffle(allPosts)
    for post in allPosts:
        world.discoveryFeed.addUpdatePost(0, post["name"], post["post"], signals=[])

    return scoringInfo



#
#   Task Scoring
#

class SmallSkillsDiscoveryFeedTask(Task):

    def __init__(self, world, scoringInfo):
        self.objectToMove = scoringInfo["objectToMove"]
        self.destinationContainer = scoringInfo["destinationContainer"]
        self.correctPostName = scoringInfo["correctPostName"]

        #taskDescription = "Your task is to follow the instructions of the other person in the room.\n"
        taskDescription = "Your task is to read the Discovery Feed, and follow the instructions posted by `" + self.correctPostName + "`, while ignoring the instructions by the other posters.\n"

        super().__init__("SmallSkillsDiscoveryFeedTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardPick = ScorecardElement("Pick up the object", "Pick up the " + self.objectToMove.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPick)

        # Reach Stage 2
        self.scorecardPlace = ScorecardElement("Place the object", "Place the " + self.objectToMove.name + " in the " + self.destinationContainer.name + ".", maxScore=1)
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

        # Check to see if the object is in the destination container
        if (not self.scorecardPlace.completed):
            # Get the object's parent container
            parentContainer = self.objectToMove.parentContainer
            if (parentContainer != None):
                if (parentContainer.uuid == self.destinationContainer.uuid):
                    self.scorecardPlace.updateScore(1, True, associatedUUIDs=[self.objectToMove.uuid, self.destinationContainer.uuid], associatedNotes=f"Agent has placed the object {self.objectToMove.uuid} in the destination container (UUID: {self.destinationContainer.uuid}).")

        # Check for a failure condition: If there's anything inside the pot other than the object to move, the task is failed.
        destinationContainerContents = self.destinationContainer.contents
        if (self.scorecardPlace.completed == False) and (len(destinationContainerContents) > 0):
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
