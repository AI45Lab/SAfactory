import random
from discoveryworld.Agent import Agent
from discoveryworld.DialogTree import DialogMaker, DialogTree, DialogNode
from discoveryworld.Layer import Layer
from discoveryworld.buildings.soil_research import mkRandomSoilNutrientsWithSetValues, mkRandomSoilNutrientsWithSetValuesEasy, mkSoilFieldControlled, mkSoilResearchBuildingChallenge
from discoveryworld.buildings.terrain import mkFenceX, mkFenceY, mkGrassFill, mkPathX, mkPathY, mkTallTree

from discoveryworld.objects import *



def makeSeedRule(randomSeed, rng, numPositiveExamples=10, numNegativeExamples=20):
    # Possible nutrients
    possibleNutrients = ["potassium", "titanium", "lithium", "thorium", "barium"]
    # Possible values
    possibleValues = [1, 2, 3]
    possibleValuesLUT = {1: "low", 2: "medium", 3: "high"}

    # Possible kinds of rules
    possibleRules = ["xor", "and", "or", "not"]

    # Output
    positiveExamples = []
    negativeExamples = []
    ruleDescription = ""
    seedRule = {}

    # Choose a rule type
    ruleType = possibleRules[randomSeed%4]
    if (ruleType == "xor"):
        # Choose two nutrients and values.  The rule is that the seed will grow if either of the two nutrients are present, but not both.
        nutrient1 = rng.choice(possibleNutrients)
        nutrient2 = rng.choice(possibleNutrients)
        while (nutrient2 == nutrient1):
            nutrient2 = rng.choice(possibleNutrients)

        value1 = rng.choice(possibleValues)
        value2 = rng.choice(possibleValues)

        ruleDescription = "The seed will grow if either " + nutrient1 + " is present at a " + possibleValuesLUT[value1] + " level, or " + nutrient2 + " is present at a " + possibleValuesLUT[value2] + " level, but not both."
        seedRule = {
            "ruleType": "xor",
            "nutrient1": nutrient1,
            "value1": value1,
            "nutrient2": nutrient2,
            "value2": value2
        }

        remainingNutrients = []
        for nutrient in possibleNutrients:
            if (nutrient != nutrient1) and (nutrient != nutrient2):
                remainingNutrients.append(nutrient)

        # Positive examples
        for i in range(0, numPositiveExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, randomly choose whether nutrient1 or nutrient2 is present
            if (i % 2 == 0):
                nutrient2Value = value2
                while (nutrient2Value == value2):
                    nutrient2Value = rng.choice(possibleValues)

                # Rule: If nutrient1 is present, and nutrient2 is not present, then the seed will grow
                dictOut[nutrient1] = value1
                dictOut[nutrient2] = nutrient2Value
                # Add to positive examples
                positiveExamples.append(dictOut)
            else:
                nutrient1Value = value1
                while (nutrient1Value == value1):
                    nutrient1Value = rng.choice(possibleValues)
                # Rule: If nutrient2 is present, and nutrient1 is not present, then the seed will grow
                dictOut[nutrient1] = nutrient1Value
                dictOut[nutrient2] = value2
                # Add to positive examples
                positiveExamples.append(dictOut)

        # Negative examples
        for i in range(0, numNegativeExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, randomly choose whether nutrient1 or nutrient2 is present
            if (i % 2 == 0):
                # Put both nutrient 1 and nutrient 2 at correct values
                # Rule: If nutrient1 is present, and nutrient2 is present, then (by XOR) the seed will NOT grow
                dictOut[nutrient1] = value1
                dictOut[nutrient2] = value2
                # Add to negative examples
                negativeExamples.append(dictOut)
            else:
                # Put both nutrient 1 and nutrient 2 at incorrect values
                # Rule: If nutrient1 is not present, and nutrient2 is not present, then the seed will NOT grow
                nutrient1Value = value1
                while (nutrient1Value == value1):
                    nutrient1Value = rng.choice(possibleValues)
                nutrient2Value = value2
                while (nutrient2Value == value2):
                    nutrient2Value = rng.choice(possibleValues)
                # Rule: If nutrient2 is present, and nutrient1 is not present, then the seed will grow
                dictOut[nutrient1] = nutrient1Value
                dictOut[nutrient2] = nutrient2Value
                # Add to negative examples
                negativeExamples.append(dictOut)

    elif (ruleType == "and"):
        # Choose two nutrients and values.  The rule is that the seed will grow if both of the two nutrients are present.
        nutrient1 = rng.choice(possibleNutrients)
        nutrient2 = rng.choice(possibleNutrients)
        while (nutrient2 == nutrient1):
            nutrient2 = rng.choice(possibleNutrients)

        value1 = rng.choice(possibleValues)
        value2 = rng.choice(possibleValues)

        ruleDescription = "The seed will grow if both " + nutrient1 + " is present at a " + possibleValuesLUT[value1] + " level, and " + nutrient2 + " is present at a " + possibleValuesLUT[value2] + " level."
        seedRule = {
            "ruleType": "and",
            "nutrient1": nutrient1,
            "value1": value1,
            "nutrient2": nutrient2,
            "value2": value2
        }

        remainingNutrients = []
        for nutrient in possibleNutrients:
            if (nutrient != nutrient1) and (nutrient != nutrient2):
                remainingNutrients.append(nutrient)

        # Positive examples
        for i in range(0, numPositiveExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Rule: If nutrient1 is present, and nutrient2 is present, then the seed will grow
            dictOut[nutrient1] = value1
            dictOut[nutrient2] = value2
            # Add to positive examples
            positiveExamples.append(dictOut)

        # Negative examples
        for i in range(0, numNegativeExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, randomly choose whether nutrient1 or nutrient2 is present
            if (i % 2 == 0):
                # Put nutrient 1 in at correct values, and nutrient 2 at incorrect
                # Rule: If nutrient1 is present, and nutrient2 is not present, then the seed will NOT grow
                dictOut[nutrient1] = value1
                nutrient2Value = value2
                while (nutrient2Value == value2):
                    nutrient2Value = rng.choice(possibleValues)
                dictOut[nutrient2] = nutrient2Value
                # Add to negative examples
                negativeExamples.append(dictOut)
            else:
                # Put nutrient 1 in at incorrect values, and nutrient 2 at correct
                # Rule: If nutrient1 is not present, and nutrient2 is present, then the seed will NOT grow
                nutrient1Value = value1
                while (nutrient1Value == value1):
                    nutrient1Value = rng.choice(possibleValues)
                dictOut[nutrient1] = nutrient1Value
                dictOut[nutrient2] = value2
                # Add to negative examples
                negativeExamples.append(dictOut)

    elif (ruleType == "or"):
        # Choose two nutrients and values.  The rule is that the seed will grow if either of the two nutrients are present.
        nutrient1 = rng.choice(possibleNutrients)
        nutrient2 = rng.choice(possibleNutrients)
        while (nutrient2 == nutrient1):
            nutrient2 = rng.choice(possibleNutrients)

        value1 = rng.choice(possibleValues)
        value2 = rng.choice(possibleValues)

        ruleDescription = "The seed will grow if either " + nutrient1 + " is present at a " + possibleValuesLUT[value1] + " level, or " + nutrient2 + " is present at a " + possibleValuesLUT[value2] + " level."
        seedRule = {
            "ruleType": "or",
            "nutrient1": nutrient1,
            "value1": value1,
            "nutrient2": nutrient2,
            "value2": value2
        }

        remainingNutrients = []
        for nutrient in possibleNutrients:
            if (nutrient != nutrient1) and (nutrient != nutrient2):
                remainingNutrients.append(nutrient)

        # Positive examples
        for i in range(0, numPositiveExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, randomly choose whether nutrient1 or nutrient2 is present
            if (i % 3 == 0):
                nutrient2Value = value2
                while (nutrient2Value == value2):
                    nutrient2Value = rng.choice(possibleValues)

                # Rule: If nutrient1 is present, and nutrient2 is not present, then the seed will grow
                dictOut[nutrient1] = value1
                dictOut[nutrient2] = nutrient2Value

            elif (i % 3 == 1):
                nutrient1Value = value1
                while (nutrient1Value == value1):
                    nutrient1Value = rng.choice(possibleValues)
                # Rule: If nutrient2 is present, and nutrient1 is not present, then the seed will grow
                dictOut[nutrient1] = nutrient1Value
                dictOut[nutrient2] = value2

            else:
                # Rule: If both nutrient1 and nutrient2 are present, then the seed will grow
                dictOut[nutrient1] = value1
                dictOut[nutrient2] = value2


            # Add to positive examples
            positiveExamples.append(dictOut)


        # Negative examples
        for i in range(0, numNegativeExamples):
            dictOut = {}
            # First do nutrients that don't matter
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, pick random values for nutrient1 and nutrient2, as long as they're not the correct values
            nutrient1Value = value1
            while (nutrient1Value == value1):
                nutrient1Value = rng.choice(possibleValues)
            nutrient2Value = value2
            while (nutrient2Value == value2):
                nutrient2Value = rng.choice(possibleValues)
            dictOut[nutrient1] = nutrient1Value
            dictOut[nutrient2] = nutrient2Value

            # Add to negative examples
            negativeExamples.append(dictOut)

    elif (ruleType == "not"):
        # Choose one nutrient and value.  The rule is that the seed will grow if the nutrient is NOT present.
        nutrient1 = rng.choice(possibleNutrients)
        value1 = rng.choice(possibleValues)

        ruleDescription = "The seed will grow if " + nutrient1 + " is NOT present at a " + possibleValuesLUT[value1] + " level."
        seedRule = {
            "ruleType": "not",
            "nutrient1": nutrient1,
            "value1": value1
        }

        remainingNutrients = []
        for nutrient in possibleNutrients:
            if (nutrient != nutrient1):
                remainingNutrients.append(nutrient)

        # Positive examples
        for i in range(0, numPositiveExamples):
            dictOut = {}
            # First do nutrients that don't matter (i.e. 'remainingNutrients')
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, make sure nutrient1 is NOT present
            nutrient1Value = value1
            while (nutrient1Value == value1):
                nutrient1Value = rng.choice(possibleValues)
            dictOut[nutrient1] = nutrient1Value
            # Add to positive examples
            positiveExamples.append(dictOut)

        # Negative examples
        for i in range(0, numNegativeExamples):
            dictOut = {}
            # First do nutrients that don't matter
            # Add random values for remaining nutrients
            for nutrient in remainingNutrients:
                dictOut[nutrient] = rng.choice(possibleValues)

            # Then, make sure nutrient1 is present
            dictOut[nutrient1] = value1

            # Add to negative examples
            negativeExamples.append(dictOut)

    # Return
    return {"ruleType": ruleType,
            "ruleDescription": ruleDescription,
            "seedRule": seedRule,
            "positiveExamples": positiveExamples,
            "negativeExamples": negativeExamples
            }


def makeScenarioPlantGrowingChallenge(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    numPlantSites = 3

    # Make the seed rule for this scenario, and the positive and negative examples
    seedRule = makeSeedRule(world.randomSeed, world.rng, numPositiveExamples=10, numNegativeExamples=10)

    # Seed:
    #SeedRequiringNutrientsRuleBased
    # Set seed rule:
    # seed.setSeedRule(seedRule)

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 5
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with grass
    mkGrassFill(world)

    # Soil research building
    mkSoilResearchBuildingChallenge(12, 8, world, seedRule["seedRule"])
    world.addTeleportLocation("Storage Building", 12+3, 8+2)

    # Primary pilot field area
    pilotFieldStartX = 6
    pilotFieldStartY = 11
    pilotFieldSizeX = 3
    pilotFieldSizeY = 6
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


    # Randomly set 9 of the tiles to have positive examples of the required soil nutrients, and 9 to have negative examples
    # Positive
    world.rng.shuffle(pilotSoilTiles)
    offset = 0
    #print("SEED RULE")
    #import json
    #print( json.dumps(seedRule, indent=4) )
    for i in range(0, 9):
        pilotSoilTiles[offset].attributes["soilNutrients"] = seedRule["positiveExamples"][i]
        #print("Positive (" + str(offset) + "): " + str(seedRule["positiveExamples"][i]))
        offset += 1

    # Add hypothesis
    scoringInfo["criticalHypotheses"].append(seedRule["ruleDescription"])
    scoringInfo["criticalQuestions"].append("Does it clearly state something equivalent to the following rule: `" + seedRule["ruleDescription"] + "`?")

    # Negative
    for i in range(0, 9):
        pilotSoilTiles[offset].attributes["soilNutrients"] = seedRule["negativeExamples"][i]
        #print("Negative (" + str(offset) + "): " + str(seedRule["negativeExamples"][i]))
        offset += 1

    # Note the pilot field soil tiles, for scoring
    scoringInfo["pilotFieldSoilTiles"] = pilotSoilTiles

    # Add seeds to the pilot field
    scoringInfo["startingSeeds"] = []
    for soilTile in pilotSoilTiles:
        seed = world.createObject("SeedRequiringNutrientsRuleBased")
        seed.setSeedRule(seedRule["seedRule"])
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
