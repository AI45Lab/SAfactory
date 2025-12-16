import random
from discoveryworld.Layer import Layer
from discoveryworld.buildings.house import mkBuildingDivided, mkBuildingOneRoom, mkTableAndChairs


def mkBarracks(x, y, world):
    # Create a building (barracks)
    #mkBuildingOneRoom(world, x=x, y=y, width=5, height=5)
    #mkBuildingOneRoom(world, x=x, y=y, width=12, height=5)
    mkBuildingDivided(world, x=x, y=y, width=8, height=7, dividerX=0, apertureX=0, dividerY=3, apertureY=1, doorX=3, signText="Barracks")

    # Add 3 beds and bedside tables (back wall)
    world.addObject(x+2, y+1, Layer.FURNITURE, world.createObject("Bed"))
    world.addObject(x+3, y+1, Layer.FURNITURE, world.createObject("TableBedside"))
    world.addObject(x+5, y+1, Layer.FURNITURE, world.createObject("Bed"))
    world.addObject(x+6, y+1, Layer.FURNITURE, world.createObject("TableBedside"))
    #world.addObject(x+8, y+1, Layer.FURNITURE, Bed(world))
    #world.addObject(x+9, y+1, Layer.FURNITURE, TableBedside(world))

    # Add 3 beds and bedside tables (middle wall)
    world.addObject(x+2, y+4, Layer.FURNITURE, world.createObject("Bed"))
    world.addObject(x+3, y+4, Layer.FURNITURE, world.createObject("TableBedside"))
    world.addObject(x+5, y+4, Layer.FURNITURE, world.createObject("Bed"))
    world.addObject(x+6, y+4, Layer.FURNITURE, world.createObject("TableBedside"))
    #world.addObject(x+8, y+4, Layer.FURNITURE, Bed(world))
    #world.addObject(x+9, y+4, Layer.FURNITURE, TableBedside(world))


    # Add a bed
    #world.addObject(x+1, y+1, Layer.FURNITURE, Bed(world))

    # Add a bedside table
    #world.addObject(x+2, y+1, Layer.FURNITURE, TableBedside(world))


def mkInfirmary(x, y, world):
    # Create a building (barracks)
    mkBuildingOneRoom(world, x=x, y=y, width=8, height=5, signText="Infirmary")

    # Add 4 beds
    world.addObject(x+1, y+1, Layer.FURNITURE, world.createObject("Bed"))
    world.addObject(x+3, y+1, Layer.FURNITURE, world.createObject("Bed"))
    #world.addObject(x+5, y+1, Layer.FURNITURE, Bed(world))
    #world.addObject(x+7, y+1, Layer.FURNITURE, Bed(world))

    # Table
    world.addObject(x+5, y+1, Layer.FURNITURE, world.createObject("Table"))
    # Fridge
    world.addObject(x+6, y+1, Layer.FURNITURE, world.createObject("Fridge"))


# Makes a mushroom that (may) have mold on it, based on the world seed
def mkMushroomScenarioAppropriate(world, seed, rng=None):
    rng = rng or random.Random()
    plant = None
    # Randomly choose one of 4 plants to turn into
    plantNames = ["mushroom1", "mushroom2", "mushroom3", "mushroom4"]
    whichPlantName = rng.choice(plantNames)
    # Then, based on the world seed, add either "", "b", "c", or "d" to the end of the plant name
    worldSeed = seed
    if (worldSeed % 5 == 0):
        whichPlantName += ""
    elif (worldSeed % 5 == 1):
        whichPlantName += "b"
    elif (worldSeed % 5 == 2):
        whichPlantName += "c"
    elif (worldSeed % 5 == 3):
        whichPlantName += "d"
    elif (worldSeed % 5 == 4):
        whichPlantName += "e"

    # Create the plant
    plant = world.createObject(whichPlantName)
    return plant

def mkCafeteria(x, y, world, rng=None):
    rng = rng or random.Random()
    # Create an L-shaped building (cafeteria)
    #mkBuildingLDivided(world, x=x, y=y, width=10, height=8, dividerX=5)
    # Create a divided building (cafeteria)
    #mkBuildingDivided(world, x=x, y=y, width=8, height=7, dividerX=0, apertureX=0, dividerY=3, apertureY=1, doorX=3, signText="Cafeteria")
    mkBuildingOneRoom(world, x=x, y=y, width=8, height=7, signText="Cafeteria")

    # Front (eating area)
    # Table and chairs
    #mkTableAndChairs(world, x=x+7, y=y+5, chairsPresent=["n", "s", "e", "w"])
    mkTableAndChairs(world, x=x+2, y=y+5, chairsPresent=["", "", "e", "w"])

    # Counter
    tables = []
    for i in range(5):
        tableToAdd = world.createObject("Table")
        #if (i == 2):
        #    tableToAdd.addObject(Mushroom(world, "red"))
        #tableToAdd.addObject(world.createObject("mushroom3"))
        #tableToAdd.addObject(world.createObject("mushroom1"))
        # Randomly choose between mushroom1 and mushroom2
        mushroom = mkMushroomScenarioAppropriate(world, world.randomSeed, rng)
        tableToAdd.addObject(mushroom)
        # if (rng.random() < 0.5):
        #     tableToAdd.addObject(world.createObject("mushroom1"))
        # else:
        #     tableToAdd.addObject(world.createObject("mushroom2"))

        world.addObject(x+i+2, y+3, Layer.FURNITURE, tableToAdd)
        tables.append(tableToAdd)

    #world.addObject(x+2, y+5, Layer.FURNITURE, Table(world))

    # Back (kitchen)
    pot = world.createObject("Pot")
    # add 5 mushrooms to pot
    #for i in range(5):
    #    pot.addObject(Mushroom(world))
    # Put the pot on a table
    kitchenPrepTable = world.createObject("Table")
    kitchenPrepTable.addObject(pot)
    world.addObject(x+3, y+1, Layer.FURNITURE, kitchenPrepTable)
    world.addObject(x+4, y+1, Layer.FURNITURE, world.createObject("Fridge"))
    world.addObject(x+5, y+1, Layer.FURNITURE, world.createObject("Sink"))
    world.addObject(x+6, y+1, Layer.FURNITURE, world.createObject("Stove"))

    # Front (decorations)
    flowerpot = world.createObject("FlowerPot")
    flowerTable = world.createObject("Table")
    flowerTable.addObject(flowerpot)
    world.addObject(x+6, y+5, Layer.FURNITURE, flowerTable)

    ## debug
    return tables, pot


def mkCafeteriaChallenge(x, y, world, rng=None):
    rng = rng or random.Random()
    # Create an L-shaped building (cafeteria)
    #mkBuildingLDivided(world, x=x, y=y, width=10, height=8, dividerX=5)
    # Create a divided building (cafeteria)
    #mkBuildingDivided(world, x=x, y=y, width=8, height=7, dividerX=0, apertureX=0, dividerY=3, apertureY=1, doorX=3, signText="Cafeteria")
    mkBuildingOneRoom(world, x=x, y=y, width=8, height=7, signText="Cafeteria")

    # Front (eating area)
    # Table and chairs
    #mkTableAndChairs(world, x=x+7, y=y+5, chairsPresent=["n", "s", "e", "w"])
    mkTableAndChairs(world, x=x+2, y=y+5, chairsPresent=["", "", "e", "w"])

    # Counter
    tables = []
    for i in range(5):
        tableToAdd = world.createObject("Table")
        #if (i == 2):
        #    tableToAdd.addObject(Mushroom(world, "red"))
        #tableToAdd.addObject(world.createObject("mushroom3"))
        #tableToAdd.addObject(world.createObject("mushroom1"))
        # Randomly choose between mushroom1 and mushroom2
        #mushroom = mkMushroomScenarioAppropriate(world, world.randomSeed, rng)
        # Create a mushroom that is itself potentially poisonous, rather than through mold/etc.
        mushroom = world.createObject("MushroomDirectlyPoisonousRandom")
        tableToAdd.addObject(mushroom)
        # if (rng.random() < 0.5):
        #     tableToAdd.addObject(world.createObject("mushroom1"))
        # else:
        #     tableToAdd.addObject(world.createObject("mushroom2"))

        world.addObject(x+i+2, y+3, Layer.FURNITURE, tableToAdd)
        tables.append(tableToAdd)

    #world.addObject(x+2, y+5, Layer.FURNITURE, Table(world))

    # Back (kitchen)
    pot = world.createObject("Pot")
    # add 5 mushrooms to pot
    #for i in range(5):
    #    pot.addObject(Mushroom(world))
    # Put the pot on a table
    kitchenPrepTable = world.createObject("Table")
    kitchenPrepTable.addObject(pot)
    world.addObject(x+3, y+1, Layer.FURNITURE, kitchenPrepTable)
    world.addObject(x+4, y+1, Layer.FURNITURE, world.createObject("Fridge"))
    world.addObject(x+5, y+1, Layer.FURNITURE, world.createObject("Sink"))
    world.addObject(x+6, y+1, Layer.FURNITURE, world.createObject("Stove"))

    # Front (decorations)
    flowerpot = world.createObject("FlowerPot")
    flowerTable = world.createObject("Table")
    flowerTable.addObject(flowerpot)
    world.addObject(x+6, y+5, Layer.FURNITURE, flowerTable)

    ## debug
    return tables, pot


def mkScienceLab(x, y, world):
    # Create a building (science lab)
    #buildingMaker.mkBuildingOneRoom(world, x=x, y=y, width=5, height=5)
    mkBuildingDivided(world, x=x, y=y, width=8, height=6, dividerX=5, apertureX=3, dividerY=0, apertureY=0, doorX=3, signText="Science Lab")
    instruments = {}
    instruments['microscope'] = world.createObject("Microscope")
    instruments['spectrometer'] = world.createObject("Spectrometer")
    instruments['phmeter'] = world.createObject("PHMeter")
    instruments['radiationmeter'] = world.createObject("RadiationMeter")
    instruments['sampler'] = world.createObject("Sampler")
    instruments['thermometer'] = world.createObject("Thermometer")
    instruments['npkmeter'] = world.createObject("NPKMeter")

    bench1 = world.createObject("Table")
    world.addObject(x+1, y+1, Layer.FURNITURE, bench1)
    #bench1.addObject( world.createObject("Microscope") )
    bench1.addObject( instruments['microscope'] )

    bench2 = world.createObject("Table")
    world.addObject(x+2, y+1, Layer.FURNITURE, bench2)
    #bench2.addObject( world.createObject("Spectrometer") )
    bench2.addObject( instruments['spectrometer'] )

    bench3 = world.createObject("Table")
    world.addObject(x+3, y+1, Layer.FURNITURE, bench3)
    #bench3.addObject( world.createObject("PHMeter") )
    bench3.addObject( instruments['phmeter'] )

    bench4 = world.createObject("Table")
    world.addObject(x+4, y+1, Layer.FURNITURE, bench4)
    #bench4.addObject( world.createObject("RadiationMeter") )
    bench4.addObject( instruments['radiationmeter'] )


    # Add sampler and sample containers (Petri dishes)
    bench5 = world.createObject("Table")
    world.addObject(x+1, y+4, Layer.FURNITURE, bench5)
    #bench5.addObject( world.createObject("Sampler") )
    bench5.addObject( instruments['sampler'] )

    bench6 = world.createObject("Table")
    world.addObject(x+1, y+3, Layer.FURNITURE, bench6)
    bench6.addObject( world.createObject("PetriDish") )


    bench7 = world.createObject("Table")
    world.addObject(x+4, y+4, Layer.FURNITURE, bench7)
    #bench7.addObject( world.createObject("Thermometer") )
    bench7.addObject( instruments['thermometer'] )


    # Add a red mushroom and a pink mushroom
    #world.addObject(x+3, y+3, Layer.OBJECTS, world.createObject("mushroom1"))
    #world.addObject(x+4, y+3, Layer.OBJECTS, world.createObject("mushroom2"))

    # Add a radioactive check source
    world.addObject(x+6, y+1, Layer.OBJECTS, world.createObject("radioactivechecksource"))

    # Add NPK meter
    #world.addObject(x+6, y+4, Layer.OBJECTS, world.createObject("NPKMeter"))
    world.addObject(x+6, y+4, Layer.FURNITURE, instruments['npkmeter'])

    return instruments


def mkStorageShed(x, y, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo):
    # Create a small building
    houseSizeX = 7
    houseSizeY = 4
    mkBuildingOneRoom(world, x=x+1, y=y, width=houseSizeX, height=houseSizeY, signText="Storage Shed", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    # Add a table in the farm house
    compoundTable1 = world.createObject("Table")
    compoundTable2 = world.createObject("Table")
    compoundTable3 = world.createObject("Table")
    compoundTable4 = world.createObject("Table")
    compoundTable5 = world.createObject("Table")

    # Create chemical dispensers
    dispenser1 = world.createObject("ChemicalDispenser")
    dispenser2 = world.createObject("ChemicalDispenser")
    dispenser3 = world.createObject("ChemicalDispenser")
    dispenser1.name = "Dispenser (Substance A)"
    dispenser2.name = "Dispenser (Substance B)"
    dispenser3.name = "Dispenser (Substance C)"
    scoringInfo['dispensers'] = [dispenser1, dispenser2, dispenser3]

    # Fill with chemicals
    #dispenser1.setAutoFill(checkObjectName="seed", fillObjectName="Seed", minCount=5)
    dispenser1.setAutoFill(checkObjectName="Substance A", fillObjectName="SubstanceA", minCount=1, replenishTime=0)
    dispenser2.setAutoFill(checkObjectName="Substance B", fillObjectName="SubstanceB", minCount=1, replenishTime=0)
    dispenser3.setAutoFill(checkObjectName="Substance C", fillObjectName="SubstanceC", minCount=1, replenishTime=0)

    # Add dispensers to tables
    compoundTable2.addObject(dispenser1)
    compoundTable3.addObject(dispenser2)
    compoundTable4.addObject(dispenser3)

    # Add bottle cleaner
    BottleCleaner = world.createObject("BottleCleaner")
    compoundTable5.addObject(BottleCleaner)
    scoringInfo['bottleCleaner'] = BottleCleaner

    # Add tables to world
    world.addObject(x+2, y+1, Layer.FURNITURE, compoundTable1)
    world.addObject(x+3, y+1, Layer.FURNITURE, compoundTable2)
    world.addObject(x+4, y+1, Layer.FURNITURE, compoundTable3)
    world.addObject(x+5, y+1, Layer.FURNITURE, compoundTable4)
    world.addObject(x+6, y+1, Layer.FURNITURE, compoundTable5)

    mixingJar = world.createObject("Jar")
    # Add to first table
    compoundTable1.addObject(mixingJar)
    scoringInfo['mixingJar'] = mixingJar

    # Add substance
    #substance1 = world.createObject("TestSubstance")
    #substance2 = world.createObject("PurpleSubstance")
    #mixingJar.addObject(substance1)
    #mixingJar.addObject(substance2)

    #substanceCleaner = world.createObject("substanceCleaner")
    #mixingJar.addObject(substanceCleaner)

    # Add rusty key
    #rustyKey = world.createObject("Key")
    rustyKey = world.createObject("KeyRustyParametric")
    rustyKey.setRustRemovalDict(chemicalSolutionDict)
    rustyKey.setKeyID(DOOR_KEY_ID)
    world.addObject(x+2, y+2, Layer.OBJECTS, rustyKey)
    scoringInfo['key'] = rustyKey

# As above but with 4 dispensers instead of 3
def mkStorageShedChallenge(x, y, world, DOOR_KEY_ID, chemicalSolutionDict, scoringInfo):
    # Create a small building
    houseSizeX = 8
    houseSizeY = 4
    mkBuildingOneRoom(world, x=x+1, y=y, width=houseSizeX, height=houseSizeY, signText="Storage Shed", includeDoor=True, doorKeyID = DOOR_KEY_ID)

    # Add a table in the farm house
    compoundTable1 = world.createObject("Table")
    compoundTable2 = world.createObject("Table")
    compoundTable3 = world.createObject("Table")
    compoundTable4 = world.createObject("Table")
    compoundTable5 = world.createObject("Table")
    compoundTable6 = world.createObject("Table")

    # Create chemical dispensers
    dispenser1 = world.createObject("ChemicalDispenser")
    dispenser2 = world.createObject("ChemicalDispenser")
    dispenser3 = world.createObject("ChemicalDispenser")
    dispenser4 = world.createObject("ChemicalDispenser")
    dispenser1.name = "Dispenser (Substance A)"
    dispenser2.name = "Dispenser (Substance B)"
    dispenser3.name = "Dispenser (Substance C)"
    dispenser4.name = "Dispenser (Substance D)"
    scoringInfo['dispensers'] = [dispenser1, dispenser2, dispenser3, dispenser4]

    # Fill with chemicals
    #dispenser1.setAutoFill(checkObjectName="seed", fillObjectName="Seed", minCount=5)
    dispenser1.setAutoFill(checkObjectName="Substance A", fillObjectName="SubstanceA", minCount=1, replenishTime=0)
    dispenser2.setAutoFill(checkObjectName="Substance B", fillObjectName="SubstanceB", minCount=1, replenishTime=0)
    dispenser3.setAutoFill(checkObjectName="Substance C", fillObjectName="SubstanceC", minCount=1, replenishTime=0)
    dispenser4.setAutoFill(checkObjectName="Substance D", fillObjectName="SubstanceD", minCount=1, replenishTime=0)

    # Add dispensers to tables
    compoundTable2.addObject(dispenser1)
    compoundTable3.addObject(dispenser2)
    compoundTable4.addObject(dispenser3)
    compoundTable5.addObject(dispenser4)

    # Add bottle cleaner
    BottleCleaner = world.createObject("BottleCleaner")
    compoundTable6.addObject(BottleCleaner)
    scoringInfo['bottleCleaner'] = BottleCleaner

    # Add tables to world
    world.addObject(x+2, y+1, Layer.FURNITURE, compoundTable1)
    world.addObject(x+3, y+1, Layer.FURNITURE, compoundTable2)
    world.addObject(x+4, y+1, Layer.FURNITURE, compoundTable3)
    world.addObject(x+5, y+1, Layer.FURNITURE, compoundTable4)
    world.addObject(x+6, y+1, Layer.FURNITURE, compoundTable5)
    world.addObject(x+7, y+1, Layer.FURNITURE, compoundTable6)

    mixingJar = world.createObject("Jar")
    # Add to first table
    compoundTable1.addObject(mixingJar)
    scoringInfo['mixingJar'] = mixingJar

    # Add substance
    #substance1 = world.createObject("TestSubstance")
    #substance2 = world.createObject("PurpleSubstance")
    #mixingJar.addObject(substance1)
    #mixingJar.addObject(substance2)

    #substanceCleaner = world.createObject("substanceCleaner")
    #mixingJar.addObject(substanceCleaner)

    # Add rusty key
    #rustyKey = world.createObject("Key")
    rustyKey = world.createObject("KeyRustyParametric")
    rustyKey.setRustRemovalDict(chemicalSolutionDict)
    rustyKey.setKeyID(DOOR_KEY_ID)
    world.addObject(x+2, y+2, Layer.OBJECTS, rustyKey)
    scoringInfo['key'] = rustyKey


def mkKeyShop(x, y, world):
    # Create a building (shop sellings colored keys)
    signText = "[Key Shop]\n<The logo is a key>"
    mkBuildingOneRoom(world, x=x, y=y, width=7, height=6, signText=signText)

    # IDX = {"g": 0, "r": 1, "b": 2, "y": 3, "w": 4, "p": 5, "o": 6}
    COLORS = {"r": "red", "g": "green", "b": "blue", "y": "yellow", "p": "pink", "w": "white", "o": "orange"}
    layout = [
        "xxxxxxx",
        "xrbywpx",
        "xxxxxxx",
        "xgxxxox",
        "xxxxxxx",
        "xxxxxxx",
    ]
    for i, row in enumerate(layout):
        for j, c in enumerate(row):
            if c in COLORS.keys():
                table = world.createObject("Table")
                world.addObject(x+j, y+i, Layer.FURNITURE, table)
                for _ in range(3):
                    key = world.createObject("ColoredKey", color=COLORS[c])
                    table.addObject(key)


def mkPaintShop(x, y, world):
    signText = "[Paint Shop]\n<The logo is a paint bucket>"
    paintShopBounds = mkBuildingOneRoom(world, x=x, y=y, width=7, height=6, signText=signText)

    #IDX = {"g": 0, "r": 1, "b": 2, "y": 3, "w": 4, "p": 5, "o": 6}
    COLORS = {"r": "red", "g": "green", "b": "blue", "y": "yellow", "p": "pink", "w": "white", "o": "orange"}
    layout = [
        "xxxxxxx",
        "xrbywpx",
        "xxxxxxx",
        "xgxxxox",
        "xxxxxxx",
        "xxxxxxx",
    ]
    colorSigns = {}
    for i, row in enumerate(layout):
        for j, c in enumerate(row):
            if c in COLORS.keys():
                table = world.createObject("TableWithSign", signText=f"[{COLORS[c]}]")
                colorSigns[COLORS[c]] = table
                world.addObject(x+j, y+i, Layer.FURNITURE, table)
                #for _ in range(IDX[c]):
                paint = world.createObject("PaintBucket", color=COLORS[c])
                table.addObject(paint)

    return colorSigns, paintShopBounds


def mkGeneralStore(x, y, world):
    # Create a building (shop sellings colored keys)
    signText = "[General Store]\n<The logo is a shopping cart>"
    mkBuildingOneRoom(world, x=x, y=y, width=9, height=14, signText=signText)

    COLORS = {"r": "red", "g": "green", "b": "blue", "y": "yellow", "p": "pink", "w": "white", "o": "orange"}
    OBJECTS = {
        "0": "",
        "1": ("Mushroom", {}),
        "2": ("Pot", {}),
        "3": ("Jar", {}),
        "4": ("Shovel", {}),
        "5": ("Seed", {}),
        "6": ("FlowerPot", {}),
        # "7": ("ColoredKey", {"color": "orange"}),
        "8": ("Flag", {}),
        "9": ("ColoredFlower", {"color": "red"}),
        # "A": "Rock",
    }
    layout = [
        ".........",
        ".0400000.",
        ".........",
        ".100.90p.",
        ".........",
        ".b02.00y.",
        ".........",
        ".g03.081.",
        ".........",
        ".040.00w.",
        ".........",
        ".5r0.60o.",
        ".........",
    ]
    for i, row in enumerate(layout):
        for j, o in enumerate(row):
            if o in COLORS.keys():
                table = world.createObject("TableWithSign", signText=f"[{COLORS[o]}] [key]")
                world.addObject(x+j, y+i, Layer.FURNITURE, table)
                for _ in range(2):
                    obj = world.createObject("ColoredKey", color=COLORS[o])
                    table.addObject(obj)

            if o == "9":
                obj = world.createObject("ColoredFlower", color="red")
                table = world.createObject("TableWithSign", signText=f"[red] [flower]")
                table.addObject(obj)

                world.addObject(x+j, y+i, Layer.FURNITURE, table)

            elif o in OBJECTS.keys():
                if OBJECTS[o]:
                    cls, kwargs = OBJECTS[o]
                    obj = world.createObject(cls, **kwargs)
                    table = world.createObject("TableWithSign", signText=f"[{obj.name}]")
                    table.addObject(obj)
                else:
                    table = world.createObject("Table")

                world.addObject(x+j, y+i, Layer.FURNITURE, table)


def mkSchool(x, y, world):
    signText = "[School]\n<The logo is a student hat>"
    schoolBounds = mkBuildingOneRoom(world, x=x, y=y, width=9, height=11, signText=signText)

    PROGRAMS = {
        0: "[Reset]",
        1: "[Add] [one]",
        2: "[Add] [two]",
        3: "[Add] [three]",
        4: "[Add] [four]",
        5: "[Add] [five]",
    }

    OBJECTS = {
        "c": ("Chair", {}),
        #"T": ("Table", {}),
        "p": ("Desk", {"facing": "south"}),
        "t": ("Desk", {"facing": "north"}),
        "<": ("Desk", {"facing": "west"}),
        ">": ("Desk", {"facing": "east"}),
        "P": ("FlagPole", {"height": 6, "current_height": 4}),
        "B": ("Bookcase", {}),
    }
    layout = [
        ".........",
        "..B...B..",
        "...TpS.P.",
        ".........",
        ".3.t.t.0.",
        ".........",
        ".t.t.1.5.",
        ".........",
        ".t.2.t.4.",
        ".........",
        ".........",
    ]

    computer = world.createObject("CountingComputer")
    measuringTape = world.createObject("MeasuringTape")
    resetDisk = None
    for i, row in enumerate(layout):
        for j, o in enumerate(row):
            if o == "P":
                cls, kwargs = OBJECTS[o]
                flagpole = world.createObject(cls, **kwargs)
                world.addObject(x+j, y+i, Layer.FURNITURE, flagpole)
                computer.flagpole = flagpole
            elif o == "S":
                desk = world.createObject("Table")
                world.addObject(x+j, y+i, Layer.FURNITURE, desk)
                desk.addObject(computer)
            elif o == "T":
                desk = world.createObject("Table")
                world.addObject(x+j, y+i, Layer.FURNITURE, desk)
                desk.addObject(measuringTape)
            elif str.isdigit(o):
                digit = int(o)
                desk = world.createObject("Desk", facing="north")
                world.addObject(x+j, y+i, Layer.FURNITURE, desk)
                disk = world.createObject("FloppyDisk", program=PROGRAMS[digit], value=digit)
                desk.addObject(disk)
                if digit == 0:
                    resetDisk = disk
            elif o in OBJECTS.keys():
                cls, kwargs = OBJECTS[o]
                obj = world.createObject(cls, **kwargs)
                world.addObject(x+j, y+i, Layer.FURNITURE, obj)

    return computer, resetDisk, measuringTape, flagpole, schoolBounds
