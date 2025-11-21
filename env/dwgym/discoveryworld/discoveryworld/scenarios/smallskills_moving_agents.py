import random

from discoveryworld.Agent import NPC, Agent
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer
from discoveryworld.Pathfinding import Pathfinder
from discoveryworld.buildings.colony import mkStorageShed, mkStorageShedChallenge
from discoveryworld.buildings.terrain import mkGrassFill, mkPathX, mkPathY, mkFenceX, mkFenceY, mkTallTree
from discoveryworld.buildings.house import mkBuildingDivided, mkBuildingOneRoom, mkTableAndChairs
from discoveryworld.DialogTree import DialogTree, DialogNode
from discoveryworld.TaskScorer import Task, ScorecardElement
from discoveryworld.Pathfinding import *


# Test whether an agent can successfully complete a scenario that just involves traversing a dialog tree
def makeScenarioMovingAgentsTest(world, numUserAgents=1):
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

    numBuildings = 3

    randomBuildingLocations = [(12, 5), (19, 5), (12, 12), (19, 12), (12, 20), (19, 20)]
    world.rng.shuffle(randomBuildingLocations)
    # Buildings
    for i in range(0, numBuildings):
        mkBuildingOneRoom(world, x=randomBuildingLocations[i][0], y=randomBuildingLocations[i][1], width=4, height=4, signText="House " + str(i), includeDoor=True, doorKeyID = 0)

    #mkBuildingOneRoom(world, x=15, y=10, width=4, height=4, signText="Talking Hut", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    # Add a table
    #compoundTable1 = world.createObject("Table")
    #world.addObject(16, 11, Layer.OBJECTS, compoundTable1)

    # Make a key for the door
    #key = world.createObject("Key", isRusted=False)
    #key.setKeyID(123)
    # Add key to table
    #compoundTable1.addObject(key)

    # Add 3 agents
    scoringInfo["agentsToTalkTo"] = []
    randomAgentLocations = randomBuildingLocations[numBuildings:]
    randomCharacterNames = ["James", "Spock", "Deanna", "Kathryn", "Benjamin", "Michael"]
    world.rng.shuffle(randomCharacterNames)
    for i in range(0, 3):
        location = randomAgentLocations[i]

        npc = NPCMovingAgentTest(world, randomCharacterNames[i], preferredX=location[0]+2, preferredY=location[1]+2, homeX=randomBuildingLocations[i][0]+2, homeY=randomBuildingLocations[i][1]+2)
        world.addObject(location[0], location[1], Layer.AGENT, npc)
        world.addAgent(npc)
        mkDialogGoHomeAgentNPC(npc, idx=i)
        scoringInfo["agentsToTalkTo"].append(npc)

    # Paths
    mkPathY(17, 2, 28, world)       # Top to bottom
    mkPathX(2, 17, 28, world)       # Left to right

    mkPathX(13, 9, 9, world)       # Left to right
    mkPathX(13, 24, 9, world)       # Left to right

    # Fence along the top
    mkFenceX(9, 2, 7, world)
    mkFenceX(19, 2, 7, world)

    # Fence along the bottom
    mkFenceX(9, 28, 7, world)
    mkFenceX(19, 28, 7, world)

    # Fence along the left side
    mkFenceY(9, 2, 13, world)
    mkFenceY(9, 19, 9, world)

    # Fence along the right side
    mkFenceY(25, 2, 13, world)
    mkFenceY(25, 19, 9, world)


    # Left side, beside houses
    mkTallTree(10+world.rng.randint(0, 1), 10+world.rng.randint(0, 1), world)
    mkTallTree(10+world.rng.randint(0, 1), 4+world.rng.randint(0, 1), world)
    mkTallTree(10+world.rng.randint(0, 1), 22+world.rng.randint(0, 1), world)
    # Right side, beside houses
    mkTallTree(23+world.rng.randint(0, 1), 10+world.rng.randint(0, 1), world)
    mkTallTree(23+world.rng.randint(0, 1), 4+world.rng.randint(0, 1), world)
    mkTallTree(23+world.rng.randint(0, 1), 22+world.rng.randint(0, 1), world)

    # Top near fence opening
    mkTallTree(13+world.rng.randint(0, 1), 3, world)
    mkTallTree(19+world.rng.randint(0, 1), 3, world)

    # Bottom near fence opening
    mkTallTree(13+world.rng.randint(0, 1), 29+world.rng.randint(0, 1), world)
    mkTallTree(19+world.rng.randint(0, 1), 29+world.rng.randint(0, 1), world)

    # Scattered trees outside the fence
    mkTallTree(2+world.rng.randint(0, 2), 2+world.rng.randint(0, 2), world)
    mkTallTree(2+world.rng.randint(0, 2), 12+world.rng.randint(0, 2), world)
    mkTallTree(2+world.rng.randint(0, 2), 22+world.rng.randint(0, 2), world)

    mkTallTree(28+world.rng.randint(0, 2), 2+world.rng.randint(0, 2), world)
    mkTallTree(28+world.rng.randint(0, 2), 12+world.rng.randint(0, 2), world)
    mkTallTree(28+world.rng.randint(0, 2), 22+world.rng.randint(0, 2), world)

    #mkTallTree(21, 8, world)
    #mkTallTree(17, 6, world)
    #mkTallTree(15, 15, world)
    #mkTallTree(18, 20, world)
    #mkTallTree(22, 11, world)
    #mkTallTree(26, 16, world)
    #mkTallTree(29, 16, world)
    #mkTallTree(13, 18, world)
    #mkTallTree(14, 16, world)
    #mkTallTree(10, 9, world)
    #mkTallTree(8, 11, world)
    #mkTallTree(7, 15, world)
    #mkTallTree(28, 8, world)

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
        world.addObject(16+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("initial location", 16, 12)


    return scoringInfo


# NPC
class NPCMovingAgentTest(NPC):
    #def __init__(self, world, name):
    def __init__(self, world, name, preferredX=15, preferredY=15, homeX=15, homeY=15):
        # Default sprite name
        super().__init__(world, name, defaultSpriteName="character32_agent_facing_east")

        # Randomly pick a sprite

        # Rendering
        self.attributes["faceDirection"] = "east"
        #self.spriteCharacterPrefix = "character32_"
        # Randomly pick a sprite
        spriteChoices = ["character32_", "character17_", "character16_", "character35_", "character9_"]
        self.spriteCharacterPrefix = world.rng.choice(spriteChoices)

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

        # Add default action (wandering), which has a low priority
        self.addAutopilotActionToQueue( AutopilotAction_Wander(preferredX=preferredX, preferredY=preferredY) )

        # Store this agent's home location
        self.attributes['homeX'] = homeX
        self.attributes['homeY'] = homeY

    # Tick
    def tick(self):
        # Stop if the object has already had tick() called this update -- this might have happened if the object moved locations in this current update cycle.
        if (self.tickCompleted):
            return

        # Debug
        print("*** NPC States (name: " + self.name + "): " + str(self.attributes['states']))

        # Call superclass
        NPC.tick(self)

        # Check for specific states
        if (("goHome_" + self.name) in self.attributes['states']):
            # Remove state
            self.attributes['states'].remove("goHome_" + self.name)
            # Clear all actions in the autopilot action queue
            self.clearAutopilotActionQueue()
            # Move to a specific location
            homeX = self.attributes['homeX']
            homeY = self.attributes['homeY']
            self.addAutopilotActionToQueue( AutopilotAction_GotoXY(x=homeX, y=homeY, priority=100) )

        # Display autopilot action queue (debug)
        print(self.displayAutopilotQueueStr())

        # Get the NPC's current autopilot action
        if (len(self.autopilotActionQueue) > 0) and (self.attributes['inDialogWith'] == None):
            # Get the current autopilot action
            curAutopilotAction = self.autopilotActionQueue[0]
            # Call the action interpreter to run it
            #print("(Agent: " + self.name + "): Calling action interpreter with action: " + str(curAutopilotAction))
            result = self.pathfinder.actionInterpreter(curAutopilotAction, agent=self, world=self.world)
            #print("(Agent: " + self.name + "): Result of calling action interpreter: " + str(result))

            # If the result is "COMPLETED", then remove the action from the queue
            if (result == ActionResult.COMPLETED):
                self.autopilotActionQueue.remove(curAutopilotAction)
                #print("(Agent: " + self.name + "): Action completed.  Removed from queue.")
            # If the result is "FAILURE", then remove the action from the queue
            elif (result == ActionResult.FAILURE):
                self.autopilotActionQueue.remove(curAutopilotAction)
                #print("(Agent: " + self.name + "): Action failed.  Removed from queue.")
            # If the result is "INVALID", then remove the action from the queue
            elif (result == ActionResult.INVALID):
                self.autopilotActionQueue.remove(curAutopilotAction)
                #print("(Agent: " + self.name + "): Action invalid.  Removed from queue.")





def mkDialogGoHomeAgentNPC(agent, idx):
    tree = DialogTree(agent)

    # Dialog has 1 stage.
    # The agent let's the NPC know that it's about to rain and they might want to go inside.
    # The NPC responds with a thank you.

    # rootNode.addDialogOption("You can't quite understand what the elder is saying, but that seems important.", "endNode", antiStates=["interestingItems"])

    # Root node
    rootNode = DialogNode("rootNode", "Hello, how can I help you?", statesToAdd=[])
    rootNode.addDialogOption("It looks like it's about to rain.  You might want to go home.", "goHomeNode", antiStates=["completed" + str(idx)])
    rootNode.addDialogOption("Oh right, I already spoke to you about the rain. Nevermind.", "endDialog", requiresStates=["completed" + str(idx)])
    rootNode.addDialogOption("Nevermind", "endDialog", antiStates=["completed" + str(idx)])
    tree.addNode(rootNode)
    tree.setRoot("rootNode")

    # Go home node
    goHomeNode = DialogNode("goHomeNode", "Thank you for letting me know.  I'll head home now.", statesToAdd=["goHome_" + agent.name, "completed" + str(idx)])
    goHomeNode.addDialogOption("Safe travels", "endDialog")
    tree.addNode(goHomeNode)

    # End dialog
    endDialog = DialogNode("endDialog", "Goodbye.")
    tree.addNode(endDialog)

    # Store dialog tree in agent
    agent.setDialogTree(tree)


#
#   Task Scoring
#
class SmallSkillsMovingAgentsTask(Task):

    def __init__(self, world, scoringInfo):
        taskDescription = "Your task is to talk to each of the 3 other characters in this environment, and warn each that it's about to rain.\n"

        super().__init__("SmallSkillsMovingAgentsTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardTalk = ScorecardElement("Talk to the other characters", "The agent has talked to all other characters.", maxScore=3)
        self.scoreCard.append(self.scorecardTalk)

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

        # Check if the agent has talked to the other person
        if not self.scorecardTalk.completed:
            completedStates = set()
            for i in range(0, 3):
                for agent in self.scoringInfo["agentsToTalkTo"]:
                    if (("completed" + str(i)) in agent.attributes["states"]):
                        completedStates.add("completed" + str(i))
            # If all agents have completed the task, then the agent has talked to the other person
            completed = False
            if (len(completedStates) == 3):
                completed = True

            # Score
            if (completed):
                self.scorecardTalk.updateScore(3, True, associatedUUIDs=[], associatedNotes=f"Agent has talked to all other characters.")
            else:
                self.scorecardTalk.updateScore(len(completedStates), False, associatedUUIDs=[], associatedNotes=f"Agent has not talked to all other characters.")

        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        self.completed = False
        self.completedSuccessfully = False
        if (self.score == self.maxScore):
            self.completed = True
            self.completedSuccessfully = True
