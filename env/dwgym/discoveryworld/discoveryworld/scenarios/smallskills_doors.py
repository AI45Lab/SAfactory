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
def makeScenarioDoorsTest(world, numUserAgents=1):
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
    startX = 15
    startY = 9
    width = 14
    height = 14
    mkBuildingOneRoom(world, x=startX, y=startY, width=width, height=height, signText="Door Dilemma", includeDoor=True, doorKeyID = 0)

    doorLocations = []

    # Make a grid of walls (x)
    yOffset1 = world.rng.randint(3, 4)
    for i in range(1, width-1):
        world.addObject(startX + i, startY + yOffset1, Layer.BUILDING, world.createObject("Wall"))

    # Make a grid of walls (x)
    yOffset2 = world.rng.randint(6, 7)
    for i in range(1, width-1):
        world.addObject(startX + i, startY + yOffset2, Layer.BUILDING, world.createObject("Wall"))

    # Make a grid of walls (x)
    yOffset3 = world.rng.randint(9, 10)
    for i in range(1, width-1):
        world.addObject(startX + i, startY + yOffset3, Layer.BUILDING, world.createObject("Wall"))

    # Make a grid of walls (y)
    xOffset1 = world.rng.randint(3, 5)
    for i in range(yOffset1, yOffset3):
        # First, check that there's no existing wall in this location
        x = startX + xOffset1
        y = startY + i
        objsAtLocation = world.getObjectsAt(x, y)
        wallsAtLocation = [obj for obj in objsAtLocation if obj.type == "wall"]
        if (len(wallsAtLocation) == 0):
            world.addObject(x, y, Layer.BUILDING, world.createObject("Wall"))

    # Doors for first column: Must be at y=yOffset1, y=yOffset2, y=yOffset3, and x=2 to xOffset1-1
    doorLocations.append((startX + world.rng.randint(1, xOffset1-1), startY + yOffset1))
    doorLocations.append((startX + world.rng.randint(1, xOffset1-1), startY + yOffset2))
    doorLocations.append((startX + world.rng.randint(1, xOffset1-1), startY + yOffset3))

    # Make a grid of walls (y)
    xOffset2 = world.rng.randint(8, 9)
    for i in range(1, yOffset2):
        # First, check that there's no existing wall in this location
        x = startX + xOffset2
        y = startY + i
        objsAtLocation = world.getObjectsAt(x, y)
        wallsAtLocation = [obj for obj in objsAtLocation if obj.type == "wall"]
        if (len(wallsAtLocation) == 0):
            world.addObject(x, y, Layer.BUILDING, world.createObject("Wall"))

    # Doors for the second column
    doorLocations.append((startX + world.rng.randint(xOffset1+1, xOffset2-1), startY + yOffset1))
    doorLocations.append((startX + world.rng.randint(xOffset1+1, xOffset2-1), startY + yOffset2))

    # Doors for the third column
    doorLocations.append((startX + world.rng.randint(xOffset2+1, width-2), startY + yOffset1))
    doorLocations.append((startX + world.rng.randint(xOffset2+1, width-2), startY + yOffset2))


    # Add doors
    scoringInfo["doors"] = []
    for doorLocation in doorLocations:
        # Remove any existing walls at the door location
        objsAtLocation = world.getObjectsAt(doorLocation[0], doorLocation[1])
        wallsAtLocation = [obj for obj in objsAtLocation if obj.type == "wall"]
        for wall in wallsAtLocation:
            world.removeObject(wall)
        # Add door
        door = world.createObject("Door")
        world.addObject(doorLocation[0], doorLocation[1], Layer.BUILDING, door)
        scoringInfo["doors"].append(door)



    objectToPickUp = world.createObject("Flag")
    scoringInfo["objectToPickUp"] = objectToPickUp
    world.addObject(startX + width-2, startY + 1, Layer.OBJECTS, objectToPickUp)

    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # TODO: ADD KEY

        # Add the agent to a specfic location
        world.addObject(startX+(width//2)+userAgentIdx, startY+height, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)

    # Small decorations
    mkTallTree(13, 13, world)
    mkTallTree(14, 16, world)
    mkTallTree(29, 18, world)
    mkTallTree(30, 14, world)
    mkTallTree(28, 7, world)
    mkTallTree(22, 8, world)
    mkTallTree(17, 8, world)
    mkTallTree(startX+(width//2)+userAgentIdx-2, startY+height, world)
    mkTallTree(startX+(width//2)+userAgentIdx+2, startY+height, world)

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

    # Add teleport locations to world
    world.addTeleportLocation("initial location", startX+(width//2), startY+height)


    return scoringInfo



#
#   Task Scoring
#

class SmallSkillsDoorsTask(Task):

    def __init__(self, world, scoringInfo):
        self.objectToPickUp = scoringInfo["objectToPickUp"]
        self.doors = scoringInfo["doors"]

        self.doorsOpened = set()

        #taskDescription = "Your task is to follow the instructions of the other person in the room.\n"
        #taskDescription = "Your task is to pick up the `" + self.objectToMove.name + "` and place it in the `" + self.destinationContainer.name + "`.\n"
        taskDescription = "Your task is to enter the building, and go through all the doors until you reach a flag.  When you see the flag, pick it up to complete the task.  Note that for the task to be considered complete, all the doors must be opened, too.\n"

        super().__init__("SmallSkillsDoorsTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardPick = ScorecardElement("Pick up the object", "Pick up the " + self.objectToPickUp.name + ".", maxScore=1)
        self.scoreCard.append(self.scorecardPick)

        # Open Doors
        self.scorecardOpenDoors = ScorecardElement("Open all doors", "Open all doors in the building.", maxScore=len(self.doors))
        self.scoreCard.append(self.scorecardOpenDoors)

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
            parentContainer = self.objectToPickUp.parentContainer
            if (parentContainer != None):
                if (parentContainer.attributes["isAgent"] == True):
                    self.scorecardPick.updateScore(1, True, associatedUUIDs=[self.objectToPickUp.uuid], associatedNotes=f"Agent has picked up the object (UUID: {self.objectToPickUp.uuid}).")

        # Look for doors that have been opened
        for door in self.doors:
            # self.attributes['isOpenPassage'] == True
            if (door.attributes["isOpenPassage"] == True):
                self.doorsOpened.add(door.uuid)

        # Check to see if all doors have been opened
        if (not self.scorecardOpenDoors.completed):
            success = False
            if (len(self.doorsOpened) == len(self.doors)):
                success = True
            totalDoors = len(self.doors)
            self.scorecardOpenDoors.updateScore(len(self.doorsOpened), success, associatedUUIDs=list(self.doorsOpened), associatedNotes=f"Agent has opened {len(self.doorsOpened)} out of {totalDoors} doors.")


        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        if (self.score == self.maxScore):
            self.completed = True
            self.completedSuccessfully = True
