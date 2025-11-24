import random
from discoveryworld.Agent import Agent, NPCChef1, NPCColonistAuto2, NPCFarmer1
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer
from discoveryworld.buildings.cave import mkCave, mkCaveChallenge
from discoveryworld.buildings.colony import mkBarracks, mkCafeteria, mkInfirmary, mkScienceLab, mkCafeteriaChallenge
from discoveryworld.buildings.farm import mkFarm, mkFarmChallenge

from discoveryworld.buildings.terrain import mkFenceX, mkFenceY, mkGrassFill, mkPathX, mkPathY, mkSignVillage, mkTownSquare


def makeScenarioTown(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []

    scoringInfo["criticalHypotheses"].append("The mushrooms that make people ill have mold on them. The mold is directly observable with the microscope, or indirectly observable by elevated spectrometer readings, particularly with a value of 1.0 on Channel 5 of the spectrometer.")
    criticalMushroom = {
        0: "red",
        1: "green",
        2: "yellow",
        3: "pink",
        4: "red and pink",
    }
    criticialMushroomColor = criticalMushroom[(world.randomSeed % 5)]
    scoringInfo["criticalHypotheses"].append("The mushrooms that make people ill have the following color(s): " + criticialMushroomColor + ".")

    scoringInfo["criticalQuestions"].append("Does it clearly state that the mushrooms that make people ill have mold on them?")
    scoringInfo["criticalQuestions"].append("Does it clearly state either that (1) the mold can be directly observed with the microscope, AND/OR (2) that the mold can be indirectly observed by elevated spectrometer readings, particularly with a value of 1.0 on Channel 5 of the spectrometer?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that the mushrooms that make people ill have the following color(s): `" + criticialMushroomColor + "`?")

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 5
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with grass
    mkGrassFill(world)
    # Randomly place a few plants (plant1, plant2, plant3)
    for i in range(0, 10):
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)


    # Buildings
    #mkHouse(4, 4, world)

    instruments = mkScienceLab(8, 21, world)
    scoringInfo["instruments"] = instruments

    mkInfirmary(19, 4, world)
    mkBarracks(19, 11, world)

    tables, pot = mkCafeteria(19, 20, world)

    mkTownSquare(16, 18, world)

    ## TODO: Add Farm?
    mushroomsAdded = mkFarm(10, 8, world, world.rng)

    # Cave
    mkCave(0, 0, world)


    # Paths
    mkPathY(17, 1, 30, world)       # Top/bottom, through town square

    mkPathX(10, 28, 15, world)       # Bottom, along cafeteria/science lab

    mkPathX(17, 19, 15, world)       # Town square to barracks

    mkPathX(17, 10, 10, world)       # Town square to infirmary

    mkPathX(1, 19, 16, world)       # Town square to farm

    # Fences
    # Top-left corner
    mkFenceY(6, 2, 16, world)
    mkFenceX(6, 2, 10, world)

    # Bottom-left corner
    mkFenceY(6, 21, 8, world)
    mkFenceX(6, 29, 10, world)

    # Bottom-right corner
    mkFenceX(19, 29, 10, world)
    mkFenceY(28, 21, 8, world)

    # Top-right corner
    mkFenceX(19, 2, 10, world)
    mkFenceY(28, 2, 16, world)


    # Add big village sign
    mkSignVillage(16, 2, world)
    mkSignVillage(16, 29, world)

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
        userAgent.addObject(world.createObject("Shovel"))
        userAgent.addObject(world.createObject("Seed"))
        # Add the agent to a specfic location
        #world.addObject(14+userAgentIdx, 14, Layer.AGENT, userAgent)      # In farm field
        world.addObject(12+userAgentIdx, 18, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("science lab", 10, 24)
    world.addTeleportLocation("cafeteria", 20, 23)
    world.addTeleportLocation("farm", 12, 13)
    world.addTeleportLocation("cave", 3, 5)
    world.addTeleportLocation("farmers field", 18, 25)
    world.addTeleportLocation("town square", 16, 18)
    world.addTeleportLocation("barracks", 18, 11)
    world.addTeleportLocation("infirmary", 22, 7)

    # currentAgent = Agent(world)
    # #world.addObject(5, 8, Layer.AGENT, currentAgent)      # Near cave
    # #world.addObject(10, 10, Layer.AGENT, currentAgent)      # Near farm
    # #world.addObject(20, 22, Layer.AGENT, currentAgent)     # In cafeteria
    # #world.addObject(10, 24, Layer.AGENT, currentAgent)     # In science lab
    # # Add tools for agent
    # currentAgent.addObject(world.createObject("Shovel"))
    # currentAgent.addObject(world.createObject("Seed"))
    # world.addAgent(currentAgent)

    # Add an NPC
    #npcColonist = NPCColonist(world, "Example NPC")
    #world.addObject(18, 25, Layer.AGENT, npcColonist)
    #world.addAgent(npcColonist)

    # Add the NPC Chef
    npcChef = NPCChef1(world, "Chef", tables=tables, pot=pot)
    world.addObject(20, 21, Layer.AGENT, npcChef)
    dialogMaker.mkDialogChef(npcChef)
    world.addAgent(npcChef)

    # Add the NPC Farmer
    npcFarmer = NPCFarmer1(world, "Farmer", mushroomsAdded)
    world.addObject(11, 12, Layer.AGENT, npcFarmer)
    dialogMaker.mkDialogFarmer(npcFarmer)
    world.addAgent(npcFarmer)


    # Add another NPC colonist
    #npcColonist1 = NPCColonist1(world, "Colonist 1", thingToPickup=None)
    ##npcColonist1 = NPCColonist1(world, "Colonist 1", thingsToPickup=mushroomsAdded, whereToPlace=pot)
    ##world.addObject(18, 20, Layer.AGENT, npcColonist1)

    # Add another NPC colonist
    npcColonists = []
    colonistPreferredWanderLocations = [
        (18, 25),   # Near farm
        (16, 18),   # Town square
        (18, 11),   # Barracks
        (13, 5),    # Near north entrance
        (8, 19)    # Near west entrance
    ]
    scoringInfo["colonists"] = []
    for i in range(0, 5):
        colonist = NPCColonistAuto2(world, "Colonist " + str(i), preferredX=colonistPreferredWanderLocations[i][0], preferredY=colonistPreferredWanderLocations[i][1])
        dialogMaker.mkDialogColonist(colonist)
        world.addObject(12+i, 19, Layer.AGENT, colonist)
        world.addAgent(colonist)

        npcColonists.append(colonist)
        scoringInfo["colonists"].append(colonist)

    return scoringInfo



#
#   Challenge version
#
def makeScenarioTownChallenge(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalHypotheses"].append("While it's unclear exactly why the mushrooms are making people ill, the glowing rock from the cave becomes luminous when it is near a harmful mushroom, providing a mechanism of detecting them.  Further, while some properties, such as color, appear to be suggestive of harmful mushrooms, they are only probabilistic, while the glowing rock is the only definitive indicator.")
    scoringInfo["criticalQuestions"] = []
    scoringInfo["criticalQuestions"].append("Does it clearly state that the glowing rock can be used as an instrument to detect harmful mushrooms, by becoming luminous when it is near a harmful mushroom?")

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 5
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with grass
    mkGrassFill(world)
    # Randomly place a few plants (plant1, plant2, plant3)
    for i in range(0, 10):
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)


    # Buildings
    #mkHouse(4, 4, world)

    instruments = mkScienceLab(8, 21, world)
    scoringInfo["instruments"] = instruments

    mkInfirmary(19, 4, world)
    mkBarracks(19, 11, world)

    tables, pot = mkCafeteriaChallenge(19, 20, world)

    mkTownSquare(16, 18, world)

    ## TODO: Add Farm?
    mushroomsAdded = mkFarmChallenge(10, 8, world, world.rng)

    # Cave
    glowingRocks = mkCaveChallenge(0, 0, world)
    scoringInfo["glowingRocks"] = glowingRocks


    # Paths
    mkPathY(17, 1, 30, world)       # Top/bottom, through town square

    mkPathX(10, 28, 15, world)       # Bottom, along cafeteria/science lab

    mkPathX(17, 19, 15, world)       # Town square to barracks

    mkPathX(17, 10, 10, world)       # Town square to infirmary

    mkPathX(1, 19, 16, world)       # Town square to farm

    # Fences
    # Top-left corner
    mkFenceY(6, 2, 16, world)
    mkFenceX(6, 2, 10, world)

    # Bottom-left corner
    mkFenceY(6, 21, 8, world)
    mkFenceX(6, 29, 10, world)

    # Bottom-right corner
    mkFenceX(19, 29, 10, world)
    mkFenceY(28, 21, 8, world)

    # Top-right corner
    mkFenceX(19, 2, 10, world)
    mkFenceY(28, 2, 16, world)


    # Add big village sign
    mkSignVillage(16, 2, world)
    mkSignVillage(16, 29, world)

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
        userAgent.addObject(world.createObject("Shovel"))
        userAgent.addObject(world.createObject("SeedDirectlyPoisonousMushroom"))
        # Add the agent to a specfic location
        #world.addObject(14+userAgentIdx, 14, Layer.AGENT, userAgent)      # In farm field
        world.addObject(12+userAgentIdx, 18, Layer.AGENT, userAgent)      # Near farm
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("science lab", 10, 24)
    world.addTeleportLocation("cafeteria", 20, 23)
    world.addTeleportLocation("farm", 12, 13)
    world.addTeleportLocation("cave", 3, 5)
    world.addTeleportLocation("farmers field", 18, 25)
    world.addTeleportLocation("town square", 16, 18)
    world.addTeleportLocation("barracks", 18, 11)
    world.addTeleportLocation("infirmary", 22, 7)

    # currentAgent = Agent(world)
    # #world.addObject(5, 8, Layer.AGENT, currentAgent)      # Near cave
    # #world.addObject(10, 10, Layer.AGENT, currentAgent)      # Near farm
    # #world.addObject(20, 22, Layer.AGENT, currentAgent)     # In cafeteria
    # #world.addObject(10, 24, Layer.AGENT, currentAgent)     # In science lab
    # # Add tools for agent
    # currentAgent.addObject(world.createObject("Shovel"))
    # currentAgent.addObject(world.createObject("Seed"))
    # world.addAgent(currentAgent)

    # Add an NPC
    #npcColonist = NPCColonist(world, "Example NPC")
    #world.addObject(18, 25, Layer.AGENT, npcColonist)
    #world.addAgent(npcColonist)

    # Add the NPC Chef
    npcChef = NPCChef1(world, "Chef", tables=tables, pot=pot)
    world.addObject(20, 21, Layer.AGENT, npcChef)
    dialogMaker.mkDialogChef(npcChef)
    world.addAgent(npcChef)

    # Add the NPC Farmer
    npcFarmer = NPCFarmer1(world, "Farmer", mushroomsAdded)
    world.addObject(11, 12, Layer.AGENT, npcFarmer)
    dialogMaker.mkDialogFarmer(npcFarmer)
    world.addAgent(npcFarmer)


    # Add another NPC colonist
    #npcColonist1 = NPCColonist1(world, "Colonist 1", thingToPickup=None)
    ##npcColonist1 = NPCColonist1(world, "Colonist 1", thingsToPickup=mushroomsAdded, whereToPlace=pot)
    ##world.addObject(18, 20, Layer.AGENT, npcColonist1)

    # Add another NPC colonist
    npcColonists = []
    colonistPreferredWanderLocations = [
        (18, 25),   # Near farm
        (16, 18),   # Town square
        (18, 11),   # Barracks
        (13, 5),    # Near north entrance
        (8, 19)    # Near west entrance
    ]
    scoringInfo["colonists"] = []
    for i in range(0, 5):
        colonist = NPCColonistAuto2(world, "Colonist " + str(i), preferredX=colonistPreferredWanderLocations[i][0], preferredY=colonistPreferredWanderLocations[i][1])
        dialogMaker.mkDialogColonist(colonist)
        world.addObject(12+i, 19, Layer.AGENT, colonist)
        world.addAgent(colonist)

        npcColonists.append(colonist)
        scoringInfo["colonists"].append(colonist)

    return scoringInfo