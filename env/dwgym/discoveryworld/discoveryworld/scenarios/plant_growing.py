import random
from discoveryworld.Agent import Agent
from discoveryworld.DialogTree import DialogMaker, DialogTree, DialogNode
from discoveryworld.Layer import Layer
from discoveryworld.buildings.soil_research import mkRandomSoilNutrientsWithSetValues, mkRandomSoilNutrientsWithSetValuesEasy, mkSoilFieldControlled, mkSoilResearchBuilding
from discoveryworld.buildings.terrain import mkFenceX, mkFenceY, mkGrassFill, mkPathX, mkPathY, mkTallTree

from discoveryworld.objects import *



def mapSeedToObjectName(nutrientName, nutrientValue):
    # List of seed types
    #seedTypes = ["SeedNutrientsPot1", "SeedNutrientsPot2", "SeedNutrientsPot3", "SeedNutrientsTit1", "SeedNutrientsTit2", "SeedNutrientsTit3", "SeedNutrientsLit1", "SeedNutrientsLit2", "SeedNutrientsLit3", "SeedNutrientsTho1", "SeedNutrientsTho2", "SeedNutrientsTho3", "SeedNutrientsBar1", "SeedNutrientsBar2", "SeedNutrientsBar3"]
    if (nutrientName == "potassium"):
        return "SeedNutrientPot" + str(nutrientValue)
    elif (nutrientName == "titanium"):
        return "SeedNutrientTit" + str(nutrientValue)
    elif (nutrientName == "lithium"):
        return "SeedNutrientLit" + str(nutrientValue)
    elif (nutrientName == "thorium"):
        return "SeedNutrientTho" + str(nutrientValue)
    elif (nutrientName == "barium"):
        return "SeedNutrientBar" + str(nutrientValue)

    print("ERROR: mapSeedToObjectName: nutrientName not recognized: " + nutrientName)
    return None


def makeScenarioPlantGrowing(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    numPlantSites = 3

    # Randomly choose what value is helpful for growing plants
    possibleNutrients = ["potassium", "titanium", "lithium", "thorium", "barium"]
    possibleValues = [1, 2, 3]
    possibleValuesStrLUT = {1: "low", 2: "medium", 3: "high"}
    whichNutrientPositive = world.rng.choice(possibleNutrients)
    whichValuePositive = world.rng.choice(possibleValues)
    whichSeedName = mapSeedToObjectName(whichNutrientPositive, whichValuePositive)

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 5
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with grass
    mkGrassFill(world)

    # Soil research building
    mkSoilResearchBuilding(12, 8, world, whichSeedName)
    world.addTeleportLocation("Storage Building", 12+3, 8+2)

    # Primary pilot field area
    pilotFieldStartX = 6
    pilotFieldStartY = 14
    pilotFieldSizeX = 4
    pilotFieldSizeY = 3
    pilotSoilTiles = []
    for i in range(0, pilotFieldSizeX):
        for j in range(0, pilotFieldSizeY):
            soilTile = world.createObject("SoilTile")
            # Randomize nutrient levels in this soil tile
            soilTile.attributes["soilNutrients"] = mkRandomSoilNutrientsWithSetValues(setValuesDict={}, rng=world.rng)
            soilTile.attributes["testField"] = False
            world.addObject(pilotFieldStartX+i, pilotFieldStartY+j, Layer.BUILDING, soilTile)
            # TODO: Add plants, etc.
            pilotSoilTiles.append(soilTile)


    # Randomly set 3 of the tiles to have positive examples of the required soil nutrients, and 3 to have negative examples
    # Positive (first 3)
    world.rng.shuffle(pilotSoilTiles)
    for i in range(0, 3):
        setValueDict = {}
        setValueDict[whichNutrientPositive] = whichValuePositive
        pilotSoilTiles[i].attributes["soilNutrients"] = mkRandomSoilNutrientsWithSetValues(setValuesDict=setValueDict, rng=world.rng)

    # Add hypothesis
    hypothesisStr = "If a seed is placed in soil with a '" + whichNutrientPositive + "' nutrient level of '" + str(possibleValuesStrLUT[whichValuePositive]) + "', then it will successfully grow into a plant."
    scoringInfo["criticalHypotheses"].append(hypothesisStr)
    scoringInfo["criticalQuestions"].append("Does it clearly state that if a seed is placed in soil with the `" + whichNutrientPositive + "` nutrient level of `" + str(possibleValuesStrLUT[whichValuePositive]) + "`, then it will grow?")

    # Negative (next 3)
    for i in range(3, 6):
        soilTile = pilotSoilTiles[i]
        setValueDict = {}
        # Select a random negative value for the nutrient (by negative, we mean not the positive value)
        setValueDict[whichNutrientPositive] = world.rng.choice(possibleValues)
        while (setValueDict[whichNutrientPositive] == whichValuePositive):
            setValueDict[whichNutrientPositive] = world.rng.choice(possibleValues)
        soilTile.attributes["soilNutrients"] = mkRandomSoilNutrientsWithSetValues(setValuesDict=setValueDict, rng=world.rng)

    # Note the pilot field soil tiles, for scoring
    scoringInfo["pilotFieldSoilTiles"] = pilotSoilTiles

    # Add seeds to the pilot field
    scoringInfo["startingSeeds"] = []
    for soilTile in pilotSoilTiles:
        seed = world.createObject(whichSeedName)
        soilTile.addObject(seed)
        scoringInfo["startingSeeds"].append(seed)

    # Sign for the pilot field
    sign = world.createObject("Sign")
    sign.setText("Pilot Field")
    world.addObject(pilotFieldStartX, pilotFieldStartY+pilotFieldSizeY, Layer.FURNITURE, sign)
    world.addTeleportLocation("Pilot Field", pilotFieldStartX, pilotFieldStartY+pilotFieldSizeY-1)

    # Make 3 test fields
    testFieldStartX = 12
    testFieldStartY = 15

    testSoilTiles = []
    for i in range(0, numPlantSites):
        x = testFieldStartX + i*5
        y = testFieldStartY
        testSoilTiles.extend(mkSoilFieldControlled(x, y, world, i+1))
        world.addTeleportLocation("Experimental Field #" + str(i+1), x, y+1)

    scoringInfo["testSoilTiles"] = testSoilTiles

    # Make path along fields
    mkPathX(6, 18, 18, world)
    # Make path to research building
    mkPathY(15, 12, 12, world)

    # Make fence around entire research station + fields
    mkFenceX(4, 6, 22, world)      # Top
    mkFenceY(4, 6, 14, world)      # Left
    mkFenceY(25, 6, 15, world)     # Right
    mkFenceX(4, 20, 10, world)     # Bottom (left)
    mkFenceX(17, 20, 9, world)     # Bottom (right)

    # Sign for the whole research facility
    sign = world.createObject("Sign")
    sign.setText("Botanical Research Facility")
    world.addObject(13, 21, Layer.FURNITURE, sign)

    # Add big trees to either side of the research facility
    mkTallTree(11, 10, world)
    mkTallTree(9, 10, world)
    mkTallTree(7, 10, world)

    mkTallTree(18, 10, world)
    mkTallTree(20, 10, world)
    mkTallTree(22, 10, world)

    mkTallTree(12, 21, world)
    mkTallTree(18, 21, world)


    # Randomly place a few decorative plants
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
        # Add the agent to a specfic location
        world.addObject(13+userAgentIdx, 14, Layer.AGENT, userAgent)      # Near center of dig site
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Should tick the environment 10 times, to make sure the pilot seeds grow and are ready for the user to view.
    for i in range(0, 10):
        world.tick()


    # Keep track of starting plants that have grown
    scoringInfo["startingPlants"] = []
    # Get all objects in the world
    allObjects = world.getAllWorldObjects()
    for obj in allObjects:
        # If the object is an instance of the "Mushroom" class, then add it to the list of starting mushrooms
        if (isinstance(obj, Mushroom)):
            scoringInfo["startingPlants"].append(obj)
        if (isinstance(obj, SoilNutrientMeter)):
            scoringInfo["soilNutrientMeter"] = obj
        if (isinstance(obj, Shovel)):
            scoringInfo["shovel"] = obj
        if (isinstance(obj, Jar)):
            scoringInfo["jar"] = obj            # Seed jar

    # Return the helpful info for scoring
    return scoringInfo


#
#   Plant Nutrients (easy/distilled)
#

def makeScenarioPlantGrowingEasy(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    numPlantSites = 1

    # Randomly choose what value is helpful for growing plants
    possibleNutrients = ["titanium", "potassium", "barium", "lithium", "thorium"]
    whichCorrect = ["potassium", "thorium", "titanium", "lithium", "barium"]
    #positiveNutrient = world.rng.choice(possibleNutrients)
    positiveNutrient = whichCorrect[world.randomSeed % 5]
    #whichSeedName = mapSeedToObjectName(whichNutrientPositive, whichValuePositive)      ## TODO -- CHECK THIS
    negativeNutrients = copy.deepcopy(possibleNutrients)
    negativeNutrients.remove(positiveNutrient)

    # Keep these in the scorer
    scoringInfo["positiveNutrient"] = positiveNutrient
    scoringInfo["negativeNutrients"] = negativeNutrients

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 1
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with grass
    mkGrassFill(world)

    # Soil research building
    #mkSoilResearchBuilding(12, 8, world, whichSeedName)
    #world.addTeleportLocation("Storage Building", 12+3, 8+2)


    # Add a soil nutrient meter
    soilMeter = world.createObject("SoilNutrientMeter")
    # Table for soil meter
    soilMeterTable = world.createObject("Table")
    soilMeterTable.addObject(soilMeter)
    world.addObject(16, 16, Layer.FURNITURE, soilMeterTable)



    # Primary pilot field area
    pilotFieldStartX = 14
    pilotFieldStartY = 14
    pilotFieldSizeX = 2
    pilotFieldSizeY = 3
    pilotSoilTiles = []
    for i in range(0, pilotFieldSizeX):
        for j in range(0, pilotFieldSizeY):
            soilTile = world.createObject("SoilTile")
            # Randomize nutrient levels in this soil tile
            soilTile.attributes["soilNutrients"] = {} #mkRandomSoilNutrientsWithSetValuesEasy(setValuesDict={}, rng=world.rng)
            world.addObject(pilotFieldStartX+i, pilotFieldStartY+j, Layer.BUILDING, soilTile)
            # TODO: Add plants, etc.
            pilotSoilTiles.append(soilTile)


    # Randomly set 2 of the tiles to have positive examples of the required soil nutrients, and 4 to have negative examples
    # Positive (first 2)
    world.rng.shuffle(pilotSoilTiles)
    for i in range(0, 2):
        # Create positive soil nutrient dictionary
        soilNutrients = {}
        for nutrient in possibleNutrients:
            soilNutrients[nutrient] = 0
        # Set the one positive nutrient
        soilNutrients[positiveNutrient] = 3
        # Set the soil nutrients
        pilotSoilTiles[i].attributes["soilNutrients"] = soilNutrients

        # TODO: Add mushrooms to these locations
        # Get tile location
        location = pilotSoilTiles[i].getWorldLocation()    # (x, y) tuple
        # Add a mushroom here
        #mushroom = world.createObject("Mushroom")
        mushroom = world.createObject("mushroom4")
        world.addObject(location[0], location[1], Layer.OBJECTS, mushroom)

    # Add hypothesis
    hypothesisStr = "If a seed is placed in soil with the '" + positiveNutrient + "' nutrient present, then it will successfully grow into a plant."
    scoringInfo["criticalHypotheses"].append(hypothesisStr)
    scoringInfo["criticalQuestions"].append("Does it clearly state that if a seed is placed in soil with the `" + positiveNutrient + "` nutrient present, then it will grow?")

    # Negative (next 4)
    for i in range(2, 6):
        soilTile = pilotSoilTiles[i]
        # Create positive soil nutrient dictionary
        soilNutrients = {}
        for nutrient in possibleNutrients:
            soilNutrients[nutrient] = 0
        # Randomly select one nutrient
        negativeNutrient = negativeNutrients[i % 4]

        # Set the one negative nutrient
        soilNutrients[negativeNutrient] = 3
        # Set the soil nutrients
        soilTile.attributes["soilNutrients"] = soilNutrients

    # Store the soil tile references for scoring
    scoringInfo["pilotFieldSoilTiles"] = pilotSoilTiles

    # Add the soil controller
    soilController = world.createObject("SoilControllerEasy")
    #soilController.setFieldNum(1, fieldTiles = fieldTiles)
    soilController.name = "soil nutrient controller"
    # TODO: Set the soil that this field controls
    scoringInfo["soilController"] = soilController

    # Put the soil controller on a table
    soilControllerTable = world.createObject("Table")
    soilControllerTable.addObject(soilController)
    world.addObject(16, 14, Layer.FURNITURE, soilControllerTable)

    # Make fence around entire research station + fields
    mkFenceX(12, 12, 6, world)      # Top
    mkFenceY(12, 12, 6, world)      # Left
    mkFenceX(12, 18, 6, world)      # Top
    mkFenceY(17, 12, 6, world)      # Left

    # Add big trees to either side of the research facility
    mkTallTree(11, 11, world)
    mkTallTree(11, 15, world)
    mkTallTree(11, 19, world)

    mkTallTree(18, 11, world)
    mkTallTree(18, 15, world)
    mkTallTree(18, 19, world)

    # Randomly place a few decorative plants
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
        # Add the agent to a specfic location
        world.addObject(16-userAgentIdx, 15, Layer.AGENT, userAgent)      # Near center of dig site
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)
    # Add initial teleport location
    world.addTeleportLocation("initial location", 16, 15)

    # Get all objects in the world
    allObjects = world.getAllWorldObjects()
    for obj in allObjects:
        # If the object is an instance of the "Mushroom" class, then add it to the list of starting mushrooms
        if (isinstance(obj, SoilNutrientMeter)):
            scoringInfo["soilNutrientMeter"] = obj

    # Return the helpful info for scoring
    return scoringInfo


#
#   Object: Soil Controller
#
from discoveryworld.Agent import NPCDevice, NPC, Agent
class SoilControllerEasy(NPCDevice):
    # Constructor
    def __init__(self, world):
        #Object.__init__(self, world, "soil controller", "soil controller", defaultSpriteName = "instruments_soil_controller")
        Agent.__init__(self, world, "soil controller", "soil controller", defaultSpriteName = "instruments_soil_controller")

        self.spriteCharacterPrefix = ""         # Disable the character prefix for this object (just use the default sprite)

        # Default attributes
        self.attributes['isUsable'] = True                       # Can this device be used with another object? (e.g. specifically through the 'use' action)
        self.attributes['isMovable'] = False                       # Can it be moved?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Set the dialog
        mkDialogSoilNutrientControllerEasy(self, self)

    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        ### TODO: CURRENTLY JUST COPIED/PASTED FROM THE SPECTROMETER.
        pass

    def tick(self):
        # Call superclass
        NPC.tick(self)
        #Object.tick(self)

        # Show the states
        print("Soil Controller States: " + str(self.attributes['states']))


    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


# Dialog tree for the soil nutrient controller
def mkDialogSoilNutrientControllerEasy(self, agent):
    tree = DialogTree(agent)

    # ["potassium", "titanium", "lithium", "thorium", "barium"]:

    # Root node (introduce the soil nutrient controller, give options to ask to change the nutrient levels)
    rootNode = DialogNode("rootNode", "Hello, I am the soil nutrient controller.  Please select which nutrient is required for plants to grow.", statesToAdd = [], statesToRemove = [])
    # Potassium
    rootNode.addDialogOption("Potassium", "selectPotassium")
    rootNode.addDialogOption("Titanium", "selectTitanium")
    rootNode.addDialogOption("Lithium", "selectLithium")
    rootNode.addDialogOption("Thorium", "selectThorium")
    rootNode.addDialogOption("Barium", "selectBarium")
    rootNode.addDialogOption("Cancel", "endNodeOK")
    tree.addNode(rootNode)
    tree.setRoot(rootNode.name)

    # Potassium
    potassiumNode = DialogNode("selectPotassium", "Selected Potassium as the nutrient required for plants to grow.", statesToAdd = ["select_potassium"], statesToRemove = [])
    potassiumNode.addDialogOption("OK", "endNodeOK")
    tree.addNode(potassiumNode)
    # Titanium
    titaniumNode = DialogNode("selectTitanium", "Selected Titanium as the nutrient required for plants to grow.", statesToAdd = ["select_titanium"], statesToRemove = [])
    titaniumNode.addDialogOption("OK", "endNodeOK")
    tree.addNode(titaniumNode)
    # Lithium
    lithiumNode = DialogNode("selectLithium", "Selected Lithium as the nutrient required for plants to grow.", statesToAdd = ["select_lithium"], statesToRemove = [])
    lithiumNode.addDialogOption("OK", "endNodeOK")
    tree.addNode(lithiumNode)
    # Thorium
    thoriumNode = DialogNode("selectThorium", "Selected Thorium as the nutrient required for plants to grow.", statesToAdd = ["select_thorium"], statesToRemove = [])
    thoriumNode.addDialogOption("OK", "endNodeOK")
    tree.addNode(thoriumNode)
    # Barium
    bariumNode = DialogNode("selectBarium", "Selected Barium as the nutrient required for plants to grow.", statesToAdd = ["select_barium"], statesToRemove = [])
    bariumNode.addDialogOption("OK", "endNodeOK")
    tree.addNode(bariumNode)

    # Exit
    endNodeOK = DialogNode("endNodeOK", "Exiting the soil nutrient controller.", statesToAdd = [], statesToRemove = [])
    tree.addNode(endNodeOK)

    # Store dialog tree in agent
    agent.setDialogTree(tree)
