import random

from discoveryworld.Agent import Agent
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer
from discoveryworld.buildings.colony import mkStorageShed, mkStorageShedChallenge
from discoveryworld.buildings.terrain import mkGrassFill, mkPathX, mkTallTree

# Randomly create a chemical combination
# numChemicals: The number of chemical possibilities (e.g. if there are 5 checmical dispensers, then numChemicals=5)
# minChemicals: The minimum number of DIFFERENT chemicals that can be in the combinations
# minAmount: The minimum number of total parts of chemicals that must be in the combination (e.g. minAmount=3 means 1 part Chemical A and 2 parts Chemical B is valid)
# maxAmount: The maximum number of total parts of chemicals that must be in the combination (e.g. maxAmount=5 means 2 part Chemical A and 3 parts Chemical B is valid)
def mkRandomChemicalCombination(rng, numChemicals:int=3, minChemicals:int=2, minAmount:int=3, maxAmount:int=3):
    # Chemical names
    chemicalNames = ["Substance A", "Substance B", "Substance C", "Substance D", "Substance E", "Substance F"]
    # Limit chemical names to numChemicals
    chemicalNames = chemicalNames[0:numChemicals]
    # Randomly shuffle the chemical names
    rng.shuffle(chemicalNames)
    # Randomly choose between minChemicals and maxAmount
    numChemicals = rng.randint(minChemicals, maxAmount-1)

    # Randomly choose the chemicals
    chemicalDict = {}
    for i in range(0, numChemicals):
        chemicalDict[chemicalNames[i]] = 1


    # Num parts to add
    totalParts = rng.randint(minAmount, maxAmount)

    # Calculate the total number of chemical parts
    while (True):
        # Count the number of parts
        sum = 0
        for key in chemicalDict:
            sum += chemicalDict[key]

        # If we have the right number of parts, we are done
        if (sum == totalParts):
            break

        # This should never happen -- but just in case, to prevent an infinite loop, if the number of parts is too high, exit
        if (sum > totalParts):
            break

        # Otherwise, we have too few parts -- randomly pick a key, and add a part
        key = rng.choice(list(chemicalDict.keys()))
        chemicalDict[key] += 1

    return chemicalDict




# TODO: Make the task generate a random combination of chemicals that works as a de-ruster on initialization (currently hardcoded to 1-part Chemical A and 2 parts Chemical C)
def makeScenarioStorageShed(world, numUserAgents=1):
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

    # Make the random solution
    chemicalSolutionDict = {}
    if (world.randomSeed == 0):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance A": 1, "Substance C": 2}
    else:
        chemicalSolutionDict = mkRandomChemicalCombination(world.rng, numChemicals=3, minChemicals=2, minAmount=3, maxAmount=4)
    scoringInfo["chemicalSolutionDict"] = chemicalSolutionDict
    #print("Chemical solution: " + str(chemicalSolutionDict))

    # Buildings
    mkStorageShed(15, 10, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo)

    # Critical Hypothesis
    #scoringInfo["criticalHypotheses"] = ["If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed."]
    # TODO: Use the chemical solution dict to generate the critical hypothesis parametrically

    # Should be of the form "X part(s) Chemical A, Y part(s) Chemical B, ..., *AND* Z part(s) Chemical C"
    mixtureStrElems = []
    for key in sorted(chemicalSolutionDict.keys()):
        if (chemicalSolutionDict[key] == 1):
            mixtureStrElems.append(str(chemicalSolutionDict[key]) + " part " + key)
        else:
            mixtureStrElems.append(str(chemicalSolutionDict[key]) + " parts " + key)
    for i in range(0, len(mixtureStrElems)):
        if (i == len(mixtureStrElems) - 1):
            mixtureStrElems[i] = "and " + mixtureStrElems[i]
        else:
            mixtureStrElems[i] = mixtureStrElems[i] + ", "
    scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of " + "".join(mixtureStrElems) + ", then the rust will be removed.")
    #scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the rust remover compound is a mixture of exactly these compounds in exactly these proportions: `" + "".join(mixtureStrElems) + "`?")

    # Paths
    mkPathX(17, 15, 15, world)       # Town square to farm

    mkTallTree(14, 10, world)
    mkTallTree(20, 8, world)
    mkTallTree(18, 6, world)
    mkTallTree(16, 15, world)
    mkTallTree(18, 20, world)
    mkTallTree(25, 11, world)
    mkTallTree(24, 16, world)
    mkTallTree(28, 16, world)
    mkTallTree(14, 18, world)
    mkTallTree(12, 16, world)
    mkTallTree(11, 9, world)
    mkTallTree(7, 11, world)
    mkTallTree(8, 15, world)
    mkTallTree(29, 8, world)

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


    # DialogMaker
    dialogMaker = DialogMaker()

    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # TODO: ADD KEY

        # Add the agent to a specfic location
        #world.addObject(14+userAgentIdx, 14, Layer.AGENT, userAgent)      # In farm field
        world.addObject(18+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("shed", 18, 12)


    return scoringInfo



def makeScenarioStorageShedChallenge(world, numUserAgents=1):
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

    # Make the random solution
    chemicalSolutionDict = {}
    if (world.randomSeed == 0):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance B": 1, "Substance C": 2}
    else:
        chemicalSolutionDict = mkRandomChemicalCombination(world.rng, numChemicals=4, minChemicals=2, minAmount=3, maxAmount=4)
    scoringInfo["chemicalSolutionDict"] = chemicalSolutionDict
    #print("Chemical solution: " + str(chemicalSolutionDict))

    # Buildings
    mkStorageShedChallenge(15, 10, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo)

    # Critical Hypothesis
    #scoringInfo["criticalHypotheses"] = ["If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed."]
    # TODO: Use the chemical solution dict to generate the critical hypothesis parametrically

    # Should be of the form "X part(s) Chemical A, Y part(s) Chemical B, ..., *AND* Z part(s) Chemical C"
    mixtureStrElems = []
    for key in sorted(chemicalSolutionDict.keys()):
        if (chemicalSolutionDict[key] == 1):
            mixtureStrElems.append(str(chemicalSolutionDict[key]) + " part " + key)
        else:
            mixtureStrElems.append(str(chemicalSolutionDict[key]) + " parts " + key)
    for i in range(0, len(mixtureStrElems)):
        if (i == len(mixtureStrElems) - 1):
            mixtureStrElems[i] = "and " + mixtureStrElems[i]
        else:
            mixtureStrElems[i] = mixtureStrElems[i] + ", "
    scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of " + "".join(mixtureStrElems) + ", then the rust will be removed.")
    #scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the rust remover compound is a mixture of exactly these compounds in exactly these proportions: `" + "".join(mixtureStrElems) + "`?")

    # Paths
    mkPathX(17, 15, 15, world)       # Town square to farm

    mkTallTree(14, 10, world)
    mkTallTree(20, 8, world)
    mkTallTree(18, 6, world)
    mkTallTree(16, 15, world)
    mkTallTree(18, 20, world)
    mkTallTree(25, 11, world)
    mkTallTree(24, 16, world)
    mkTallTree(28, 16, world)
    mkTallTree(14, 18, world)
    mkTallTree(12, 16, world)
    mkTallTree(11, 9, world)
    mkTallTree(7, 11, world)
    mkTallTree(8, 15, world)
    mkTallTree(29, 8, world)

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


    # DialogMaker
    dialogMaker = DialogMaker()

    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # TODO: ADD KEY

        # Add the agent to a specfic location
        #world.addObject(14+userAgentIdx, 14, Layer.AGENT, userAgent)      # In farm field
        world.addObject(18+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("shed", 18, 12)


    return scoringInfo


#
#   Easy version
#
# TODO: Make the task generate a random combination of chemicals that works as a de-ruster on initialization (currently hardcoded to 1-part Chemical A and 2 parts Chemical C)
def makeScenarioStorageShedEasyDistilled(world, numUserAgents=1):
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

    # Make the random solution
    chemicalSolutionDict = {}
    if ((world.randomSeed % 5) == 0):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance B": 1}
    elif ((world.randomSeed % 5) == 1):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance D": 1}
    elif ((world.randomSeed % 5) == 2):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance C": 1}
    elif ((world.randomSeed % 5) == 3):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance A": 1}
    elif ((world.randomSeed % 5) == 4):
        # Hardcode the first version of the task
        chemicalSolutionDict = {"Substance C": 1}
    scoringInfo["chemicalSolutionDict"] = chemicalSolutionDict
    #print("Chemical solution: " + str(chemicalSolutionDict))

    # Buildings
    #mkStorageShed(15, 10, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo)
    mkStorageShedChallenge(15, 10, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo)

    # Critical Hypothesis
    #scoringInfo["criticalHypotheses"] = ["If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed."]
    # TODO: Use the chemical solution dict to generate the critical hypothesis parametrically

    solutionKey = list(chemicalSolutionDict.keys())[0]
    scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of pure " + str(solutionKey) + ", then the rust will be removed.")
    #scoringInfo["criticalHypotheses"].append("If the key is placed in a mixture of 1 part Chemical A and 2 parts Chemical C, then the rust will be removed.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the rust remover compound is this compound: `" + str(solutionKey) + "`?")

    # Paths
    mkPathX(17, 15, 15, world)       # Town square to farm

    mkTallTree(14, 10, world)
    mkTallTree(20, 8, world)
    mkTallTree(18, 6, world)
    mkTallTree(16, 15, world)
    mkTallTree(18, 20, world)
    mkTallTree(25, 11, world)
    mkTallTree(24, 16, world)
    mkTallTree(28, 16, world)
    mkTallTree(14, 18, world)
    mkTallTree(12, 16, world)
    mkTallTree(11, 9, world)
    mkTallTree(7, 11, world)
    mkTallTree(8, 15, world)
    mkTallTree(29, 8, world)

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


    # DialogMaker
    dialogMaker = DialogMaker()

    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # TODO: ADD KEY

        # Add the agent to a specfic location
        #world.addObject(14+userAgentIdx, 14, Layer.AGENT, userAgent)      # In farm field
        world.addObject(18+userAgentIdx, 12, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("shed", 18, 12)


    return scoringInfo
