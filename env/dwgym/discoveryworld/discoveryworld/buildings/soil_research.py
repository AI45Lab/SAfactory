

# Make a field of soil that can be controlled by a soil nutrient manager
from discoveryworld.Layer import Layer
from discoveryworld.buildings.house import mkBuildingOneRoom


def mkSoilFieldControlled(x, y, world, fieldNumber, width=2, height=2):
    # Create the field
    fieldTiles = []
    for i in range(0, width):
        for j in range(0, height):
            soilTile = world.createObject("SoilTile")
            # Set a baseline soil nutrient level
            nutrientLevels = packSoilNutrients(potassium=0, titanium=0, lithium=0, thorium=0, barium=0)
            soilTile.attributes["soilNutrients"] = nutrientLevels
            soilTile.attributes["testField"] = True
            fieldTiles.append(soilTile)     # Keep track of the soil tiles, so we can let the soil nutrient manager know which tiles it controls

            world.addObject(x+i, y+j, Layer.BUILDING, soilTile)

    # Add a sign saying what field this is
    sign = world.createObject("Sign")
    sign.setText("Soil Field #" + str(fieldNumber))
    world.addObject(x, y+2, Layer.FURNITURE, sign)

    # Add the soil controller
    soilController = world.createObject("SoilController")
    soilController.setFieldNum(fieldNumber, fieldTiles = fieldTiles)
    soilController.name = "soil nutrient controller (field #" + str(fieldNumber) + ")"
    # TODO: Set the soil that this field controls

    # Put the soil controller on a table
    soilControllerTable = world.createObject("Table")
    soilControllerTable.addObject(soilController)
    world.addObject(x+1, y+2, Layer.FURNITURE, soilControllerTable)

    return fieldTiles


def mkSoilResearchBuilding(x, y, world, whichSeedName):
    # Create a small building
    houseSizeX = 4
    houseSizeY = 4
    mkBuildingOneRoom(world, x=x+1, y=y, width=houseSizeX, height=houseSizeY, signText="Storage Building")

    # Add a seed jar
    seedJar = world.createObject("Jar")
    #seedJar.setAutoFill(checkObjectName="seed", fillObjectName="Seed", minCount=5)
    seedJar.setAutoFill(checkObjectName="seed", fillObjectName=whichSeedName, minCount=1, replenishTime=0)
    seedJar.name = "seed jar"

    # Table for seed jar
    seedTable = world.createObject("Table")
    seedTable.addObject(seedJar)
    world.addObject(x+3, y+1, Layer.FURNITURE, seedTable)

    # Add a soil nutrient meter
    soilMeter = world.createObject("SoilNutrientMeter")
    # Table for soil meter
    soilMeterTable = world.createObject("Table")
    soilMeterTable.addObject(soilMeter)
    world.addObject(x+2, y+1, Layer.FURNITURE, soilMeterTable)

    # Add a shovel in the farm house
    shovel = world.createObject("Shovel")
    world.addObject(x+2, y+2, Layer.FURNITURE, shovel)


## TODO: SET SEED RULE!
def mkSoilResearchBuildingChallenge(x, y, world, seedRule):
    # Create a small building
    houseSizeX = 4
    houseSizeY = 4
    mkBuildingOneRoom(world, x=x+1, y=y, width=houseSizeX, height=houseSizeY, signText="Storage Building")

    # Add a seed jar
    seedJar = world.createObject("Jar")
    #seedJar.setAutoFill(checkObjectName="seed", fillObjectName="Seed", minCount=1, replenishTime=0)  # Not using autofill, just populating with 10 static seeds due to the special requirements for each seed
    seedJar.name = "seed jar"

    # Add 10 seeds
    for i in range(0, 10):
        seed = world.createObject("SeedRequiringNutrientsRuleBased")
        seed.setSeedRule(seedRule)
        seedJar.addObject(seed)


    # Table for seed jar
    seedTable = world.createObject("Table")
    seedTable.addObject(seedJar)
    world.addObject(x+3, y+1, Layer.FURNITURE, seedTable)

    # Add a soil nutrient meter
    soilMeter = world.createObject("SoilNutrientMeter")
    # Table for soil meter
    soilMeterTable = world.createObject("Table")
    soilMeterTable.addObject(soilMeter)
    world.addObject(x+2, y+1, Layer.FURNITURE, soilMeterTable)

    # Add a shovel in the farm house
    shovel = world.createObject("Shovel")
    world.addObject(x+2, y+2, Layer.FURNITURE, shovel)


# Packing soil nutrients (potassium, titanium, lithium, thorium, barium)
def packSoilNutrients(potassium, titanium, lithium, thorium, barium):
    packed = {
        "potassium": potassium,
        "titanium": titanium,
        "lithium": lithium,
        "thorium": thorium,
        "barium": barium
    }
    return packed

def packSoilNutrientsList(nutrients:list):
    packed = {
        "potassium": nutrients[0],
        "titanium": nutrients[1],
        "lithium": nutrients[2],
        "thorium": nutrients[3],
        "barium": nutrients[4]
    }
    return packed

# Randomly set the soil nutrients for each parameter
def mkRandomSoilNutrientsWithSetValues(setValuesDict:dict, rng):
    # Randomly set the soil nutrients for each parameter
    out = {}
    possibleValues = [1, 2, 3]
    for nutrientName in ["potassium", "titanium", "lithium", "thorium", "barium"]:
        out[nutrientName] = rng.choice(possibleValues)

    # Then, for any nutrient that is in the setValuesDict, set it to the value in the setValuesDict
    for nutrientName in setValuesDict:
        out[nutrientName] = setValuesDict[nutrientName]

    return out

# Randomly set the soil nutrients for each parameter
def mkRandomSoilNutrientsWithSetValuesEasy(setValuesDict:dict, rng):
    # Randomly set the soil nutrients for each parameter
    out = {}
    possibleValues = [0, 1]
    for nutrientName in ["potassium", "titanium", "lithium", "thorium", "barium"]:
        out[nutrientName] = rng.choice(possibleValues)

    # Then, for any nutrient that is in the setValuesDict, set it to the value in the setValuesDict
    for nutrientName in setValuesDict:
        out[nutrientName] = setValuesDict[nutrientName]

    return out
