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
def makeScenarioDialogTest(world, numUserAgents=1):
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
    mkBuildingOneRoom(world, x=15, y=10, width=4, height=4, signText="Talking Hut", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    # Add a table
    compoundTable1 = world.createObject("Table")
    world.addObject(16, 11, Layer.OBJECTS, compoundTable1)

    # Make a key for the door
    key = world.createObject("Key", isRusted=False)
    key.setKeyID(123)
    # Add key to table
    compoundTable1.addObject(key)

    # Add the other agent
    npc = NPCDialogTest(world, "Sally")
    world.addObject(17, 11, Layer.AGENT, npc)
    world.addAgent(npc)
    mkDialogDialogTestNPC(npc, world.rng)
    scoringInfo["agentToTalkTo"] = npc

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
        world.addObject(16+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("initial location", 16, 12)


    return scoringInfo


# NPC
class NPCDialogTest(NPC):
    def __init__(self, world, name):
        # Default sprite name
        super().__init__(world, name, defaultSpriteName="character32_agent_facing_east")

        # Rendering
        self.attributes["faceDirection"] = "east"
        self.spriteCharacterPrefix = "character32_"

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


def mkDialogDialogTestNPC(agent, rng):
    tree = DialogTree(agent)

    # Dialog will have 3 stages, each with 8 options.
    # In the first stage, the agent will have to select the word "apple" from other fruit names.
    # In the second stage, the agent will have to select the word "car" from other vehicle names.
    # In the third stage, the agent will have to select the word "tree" from other plant names.
    # If the agent selects the correct word all 3 times, it'll set the state "taskCompleted".
    # If the agent selects the wrong word at any stage, it'll go to an exit node, where the agent will stop talking to the NPC.

    # First stage (root node)
    optionsStage1 = ["banana", "orange", "grape", "pear", "apple", "kiwi", "peach", "lime"]
    correctOptionStage1 = rng.choice(optionsStage1)
    rootNode = DialogNode("rootNode", f"I'd like you to select the option that says `" + correctOptionStage1 + "`.", statesToAdd=["stage1"])
    # Shuffle the options
    rng.shuffle(optionsStage1)
    for option in optionsStage1:
        rootNode.addDialogOption(option, "endNodeFailure" if option != correctOptionStage1 else "stage2Node")
    rootNode.addDialogOption("Exit", "endDialog")
    tree.addNode(rootNode)
    tree.setRoot(rootNode.name)

    # Second stage
    optionsStage2 = ["bus", "truck", "bike", "motorcycle", "car", "scooter", "van", "train"]
    correctOptionStage2 = rng.choice(optionsStage2)
    stage2Node = DialogNode("stage2Node", f"Great! Now, select the option that says `" + correctOptionStage2 + "`.", statesToAdd=["stage2"])
    # Shuffle the options
    rng.shuffle(optionsStage2)
    for option in optionsStage2:
        stage2Node.addDialogOption(option, "endNodeFailure" if option != correctOptionStage2 else "stage3Node")
    stage2Node.addDialogOption("Exit", "endDialog")
    tree.addNode(stage2Node)

    # Third stage
    optionsStage3 = ["bush", "flower", "tree", "grass", "shrub", "weed", "fern", "moss"]
    correctOptionStage3 = rng.choice(optionsStage3)
    stage3Node = DialogNode("stage3Node", f"Great! Now, select the option that says `" + correctOptionStage3 + "`.", statesToAdd=["stage3"])
    # Shuffle the options
    rng.shuffle(optionsStage3)
    for option in optionsStage3:
        stage3Node.addDialogOption(option, "endNodeFailure" if option != correctOptionStage3 else "endNodeSuccess")
    stage3Node.addDialogOption("Exit", "endDialog")
    tree.addNode(stage3Node)

    # Exit dialog
    endDialog = DialogNode("endDialog", "Thank you for talking with me.")
    tree.addNode(endDialog)

    # End node (task failure)
    endNodeFailure = DialogNode("endNodeFailure", "I'm sorry, but that's not the correct answer.")
    endNodeFailure.addDialogOption("Exit", "endDialog")
    tree.addNode(endNodeFailure)

    # End node (task success)
    endNodeSuccess = DialogNode("endNodeSuccess", "Thank you for successfully following those instructions.", statesToAdd=["taskCompleted"])
    endNodeSuccess.addDialogOption("Exit", "endDialog")
    tree.addNode(endNodeSuccess)

    # Store dialog tree in agent
    agent.setDialogTree(tree)


#
#   Task Scoring
#
class SmallSkillsDialogTask(Task):

    def __init__(self, world, scoringInfo):
        taskDescription = "Your task is to follow the instructions of the other person in the room.\n"

        super().__init__("SmallSkillsDialogTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardTalk = ScorecardElement("Talk to other person", "The agent has talked to the other person.", maxScore=1)
        self.scoreCard.append(self.scorecardTalk)

        # Reach Stage 2
        self.scorecardStage2 = ScorecardElement("Reach Stage 2", "The agent has reached the second stage of the dialog tree.", maxScore=1)
        self.scoreCard.append(self.scorecardStage2)

        # Reach Stage 3
        self.scorecardStage3 = ScorecardElement("Reach Stage 3", "The agent has reached the third stage of the dialog tree.", maxScore=1)
        self.scoreCard.append(self.scorecardStage3)

        # Taking critical objects
        self.scorecardComplete = ScorecardElement("Complete Task", "The agent has completed going through the dialog tree successfully.", maxScore=1)
        self.scoreCard.append(self.scorecardComplete)

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
            otherAgent = self.scoringInfo["agentToTalkTo"]
            if ("stage1" in otherAgent.attributes["states"]):
                self.scorecardTalk.updateScore(1, True, associatedUUIDs=[otherAgent.uuid], associatedNotes=f"Agent has talked to the other person (UUID: {otherAgent.uuid}).")

        # Check if the agent has reached stage 2
        if not self.scorecardStage2.completed:
            otherAgent = self.scoringInfo["agentToTalkTo"]
            if ("stage2" in otherAgent.attributes["states"]):
                self.scorecardStage2.updateScore(1, True, associatedUUIDs=[otherAgent.uuid], associatedNotes=f"Agent has reached stage 2 of the dialog tree (UUID: {otherAgent.uuid}).")

        # Check if the agent has reached stage 3
        if not self.scorecardStage3.completed:
            otherAgent = self.scoringInfo["agentToTalkTo"]
            if ("stage3" in otherAgent.attributes["states"]):
                self.scorecardStage3.updateScore(1, True, associatedUUIDs=[otherAgent.uuid], associatedNotes=f"Agent has reached stage 3 of the dialog tree (UUID: {otherAgent.uuid}).")


        # Check if the agent has completed the task
        if not self.scorecardComplete.completed:
            otherAgent = self.scoringInfo["agentToTalkTo"]
            if ("taskCompleted" in otherAgent.attributes["states"]):
                self.scorecardComplete.updateScore(1, True, associatedUUIDs=[otherAgent.uuid], associatedNotes=f"Agent has completed the dialog tree (UUID: {otherAgent.uuid}).")

        # TODO: Check if the agent completed the dialog with the other person


        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        self.completed = False
        self.completedSuccessfully = False
        if (self.score == self.maxScore):
            self.completed = True
            self.completedSuccessfully = True
