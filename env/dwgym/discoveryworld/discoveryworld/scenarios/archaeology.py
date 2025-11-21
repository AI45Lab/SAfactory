
import random

import numpy as np
from discoveryworld.Agent import Agent
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer

from discoveryworld.buildings import mkGrassFill
from discoveryworld.buildings.archaeology import mkDigSite, mkDigSiteWithObj
from discoveryworld.buildings.house import mkBuildingOneRoom
from discoveryworld.buildings.terrain import mkFenceX, mkFenceY, mkGrassFill, mkPathX, mkPathY, mkTallTree


# This is the SIMPLE version of the task, with a radiocarbon meter
def makeScenarioArchaeologicalDig(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    numDigSites = 3

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

    # Possible artifact ages (in years). Ranges from 100k to 1 million years old, with even numbers.
    artifactAges = [150000, 230000, 370000, 410000, 500000, 675000, 725000, 890000, 930000, 1000000]
    # Shuffle the artifact ages
    world.rng.shuffle(artifactAges)
    # Trim to the first numDigSites
    artifactAges = artifactAges[:numDigSites]
    # Find the oldest artifact
    oldestArtifactAge = max(artifactAges)

    # List dig site locations
    digSiteLocations = [(10, 10), (20, 13), (12, 18)]

    # Add 3 dig sites
    scoringInfo["unknownArtifacts"] = []
    scoringInfo["signs"] = []
    otherArtifactAges = []
    goldDigSiteIdx = -1
    for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
        artifact, sign = mkDigSite(digSiteLocation[0], digSiteLocation[1], world, world.rng, digSiteIdx+1, artifactAges[digSiteIdx])
        scoringInfo["unknownArtifacts"].append(artifact)
        scoringInfo["signs"].append(sign)
        if (artifact.attributes["radiocarbonAge"] == oldestArtifactAge):
            scoringInfo["targetSign"] = sign
            goldDigSiteIdx = digSiteIdx+1
        else:
            otherArtifactAges.append(artifactAges[digSiteIdx])

    # TODO: Critical hypotheses
    #scoringInfo["criticalHypotheses"] = ["TODO: Add critical hypotheses here"]

    scoringInfo["criticalHypotheses"].append("The artifact at Dig Site #" + str(goldDigSiteIdx) + " is the oldest, with an age of " + str(oldestArtifactAge) + " years. The ages of the other artifacts (in years) are: " + ", ".join([str(age) for age in otherArtifactAges]) + ".")
    scoringInfo["criticalQuestions"].append("Does it clearly state that the artifact at Dig Site #" + str(goldDigSiteIdx) + " is the oldest, because it's age is " + str(oldestArtifactAge) + " years?")
    scoringInfo["criticalQuestions"].append("Does it clearly state the ages of the other younger artifacts (in years), which are: `" + ", ".join([str(age) for age in otherArtifactAges]) + "`?")


    # Add a table at the start of the dig site
    instrumentTable = world.createObject("Table")
    world.addObject(15, 15, Layer.FURNITURE, instrumentTable)
    # Add a radiocarbon meter to the table
    radioCarbonMeter = world.createObject("RadioCarbonMeter")
    instrumentTable.addObject( radioCarbonMeter )
    scoringInfo["radioCarbonMeter"] = radioCarbonMeter

    # Add a shovel at the start of the dig site
    shovel = world.createObject("Shovel")
    world.addObject(14, 15, Layer.FURNITURE, shovel)
    scoringInfo["shovel"] = shovel

    # Add a flag at the start of the dig site
    flag = world.createObject("Flag")
    world.addObject(14, 16, Layer.FURNITURE, flag)
    scoringInfo["flag"] = flag

    # Add some random holes
    minHoles = 1
    holeCount = 0
    while (holeCount < minHoles):
        # Pick a random locatio between 10-20
        randX = world.rng.randint(10, 20)
        randY = world.rng.randint(10, 20)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a hole
                soilTile = world.createObject("SoilTile")
                #soilTile.attributes['hasHole'] = True
                # Remove all soil tile contents (i.e. the dirt) -- this has the effect of making a hole
                for obj in soilTile.contents:
                    soilTile.removeObject(obj)

                world.addObject(randX, randY, Layer.BUILDING, soilTile)
                holeCount += 1


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


    # Add some trees (statically placed)
    mkTallTree(14, 7, world)
    mkTallTree(18, 8, world)
    mkTallTree(8, 12, world)
    mkTallTree(19, 15, world)
    mkTallTree(10, 20, world)
    mkTallTree(20, 19, world)

    # Add some small plants (randomly placed)
    plantCount = 0
    minPlants = 15
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(2, world.sizeX - 2)
        randY = world.rng.randint(2, world.sizeY - 2)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a plant
                world.addObject(randX, randY, Layer.OBJECTS, world.createObject("PlantRandomSmall"))
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


    # Add teleport locations to world
    world.addTeleportLocation("base camp", 13, 14)
    for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
        world.addTeleportLocation("dig site " + str(digSiteIdx+1), digSiteLocation[0]-1, digSiteLocation[1]+1)


    # Return the scoring info
    return scoringInfo


#
# Make the archaeological dig scenario
#

# This is the HARD version of the task, with a generic radioisotope meter, where the agent has to figure out which radioisotope is useful for dating the artifacts
def makeScenarioArchaeologicalDigGenericRadioisotope(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []

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

    # Possible artifact ages (in years). Ranges from 100k to 1 million years old, with even numbers.
    #artifactAges = [150000, 230000, 370000, 410000, 500000, 675000, 725000, 890000, 930000, 1000000]
    # Shuffle the artifact ages
    #rng.shuffle(artifactAges)
    # Trim to the first numDigSites
    #artifactAges = artifactAges[:numDigSites]

    # List dig site locations
    #seedSiteLocations = [(10, 10), (20, 13), (12, 18)]
    #experimentalSiteLocations = [(16, 8), (18, 18), (8, 15)]
    digSiteLocations = [(10, 10), (20, 13), (12, 18)] + [(16, 8), (18, 18), (8, 15)]
    # Shuffle
    world.rng.shuffle(digSiteLocations)

    #"ArtifactStoneHammer": ArtifactStoneHammer,
    #"ArtifactBronzeChisel": ArtifactBronzeChisel,
    #"ArtifactIronTongs": ArtifactIronTongs,
    knownArtifactAges = [10000, 4000, 1000]
    oldArtifactAge = world.rng.choice([40000, 35000, 30000, 20000])
    mediumArtifactAge = world.rng.choice([5000, 4000, 3000, 2000])
    youngArtifactAge = world.rng.choice([1700, 1500, 1200])

    #print("oldArtifactAge: " + str(oldArtifactAge))
    #print("mediumArtifactAge: " + str(mediumArtifactAge))
    #print("youngArtifactAge: " + str(youngArtifactAge))

    # Make up 4 faux radioisotope values.  For the real one, it should strongly correlate with the age of the artifact.  For the others, the correlation should be very weak.
    # The real one should be the first one.
    realRadioisotopeValues = knownArtifactAges + [oldArtifactAge, mediumArtifactAge, youngArtifactAge]
    # divide each value by 50,000
    realRadioisotopeValues = [val/50000 for val in realRadioisotopeValues]
    # Add some noise (+/- 0.01) to each value
    realRadioisotopeValues = [val + world.rng.uniform(-0.01, 0.01) for val in realRadioisotopeValues]
    # Take 1-the value
    realRadioisotopeValues = [1-val for val in realRadioisotopeValues]
    # Round to 3 decimal places
    realRadioisotopeValues = [round(val, 3) for val in realRadioisotopeValues]

    # Use numpy to calculate the correlation between the real radioisotope values and the artifact ages
    correlation = np.corrcoef(knownArtifactAges + [oldArtifactAge, mediumArtifactAge, youngArtifactAge], realRadioisotopeValues)
    #print ("Ages: " + str(knownArtifactAges + [oldArtifactAge, mediumArtifactAge, youngArtifactAge]))
    #print ("Real radioisotope values: " + str(realRadioisotopeValues))
    #print ("Correlation: " + str(correlation[0][1]))

    # For the other radioisotope values, just make them random, between 0 and 1

    # Make a fake radioisotope value that has a weak correlation with the artifact ages
    done = False
    fakeRadioisotope1Values = []
    while (not done):
        fakeRadioisotope1Values = [world.rng.uniform(0, 1) for i in range(0, len(realRadioisotopeValues))]
        # Round to 3 decimal places
        fakeRadioisotope1Values = [round(val, 3) for val in fakeRadioisotope1Values]
        correlation = np.corrcoef(knownArtifactAges, fakeRadioisotope1Values[:len(knownArtifactAges)])
        if (abs(correlation[0][1]) < 0.1):
            done = True
    #print("Fake radioisotope 1 values: " + str(fakeRadioisotope1Values))
    #print("Correlation: " + str(correlation[0][1]))

    done = False
    fakeRadioisotope2Values = []
    while (not done):
        fakeRadioisotope2Values = [world.rng.uniform(0, 1) for i in range(0, len(realRadioisotopeValues))]
        # Round to 3 decimal places
        fakeRadioisotope2Values = [round(val, 3) for val in fakeRadioisotope2Values]
        correlation = np.corrcoef(knownArtifactAges, fakeRadioisotope2Values[:len(knownArtifactAges)])
        if (abs(correlation[0][1]) < 0.1):
            done = True
    #print("Fake radioisotope 2 values: " + str(fakeRadioisotope2Values))
    #print("Correlation: " + str(correlation[0][1]))

    done = False
    fakeRadioisotope3Values = []
    while (not done):
        fakeRadioisotope3Values = [world.rng.uniform(0, 1) for i in range(0, len(realRadioisotopeValues))]
        # Round to 3 decimal places
        fakeRadioisotope3Values = [round(val, 3) for val in fakeRadioisotope3Values]
        correlation = np.corrcoef(knownArtifactAges + [oldArtifactAge, mediumArtifactAge, youngArtifactAge], fakeRadioisotope3Values)
        if (abs(correlation[0][1]) < 0.1):
            done = True
    #print("Fake radioisotope 3 values: " + str(fakeRadioisotope3Values))
    #print("Correlation: " + str(correlation[0][1]))
    #exit(1)

    # Figure out which channel (1, 2, 3, or 4) should be the "real" one
    channelShift = world.rng.choice([0, 1, 2, 3])

    # Assign the radioisotope values to the artifacts
    seedOldArtifact = world.createObject("ArtifactStoneHammer")
    radioisotopeValues = [realRadioisotopeValues[0], fakeRadioisotope1Values[0], fakeRadioisotope2Values[0], fakeRadioisotope3Values[0]]
    radioisotopeValues = radioisotopeValues[-channelShift:] + radioisotopeValues[:-channelShift]        # Shift the list so that the real radioisotope value is in the real channel
    seedOldArtifact.attributes["radioisotopeValues"] = radioisotopeValues
    seedOldArtifact.attributes["radiocarbonAge"] = knownArtifactAges[0]

    seedMediumArtifact = world.createObject("ArtifactBronzeChisel")
    radioisotopeValues = [realRadioisotopeValues[1], fakeRadioisotope1Values[1], fakeRadioisotope2Values[1], fakeRadioisotope3Values[1]]
    radioisotopeValues = radioisotopeValues[-channelShift:] + radioisotopeValues[:-channelShift]        # Shift the list so that the real radioisotope value is in the real channel
    seedMediumArtifact.attributes["radioisotopeValues"] = radioisotopeValues
    seedMediumArtifact.attributes["radiocarbonAge"] = knownArtifactAges[1]

    seedYoungArtifact = world.createObject("ArtifactIronTongs")
    radioisotopeValues = [realRadioisotopeValues[2], fakeRadioisotope1Values[2], fakeRadioisotope2Values[2], fakeRadioisotope3Values[2]]
    radioisotopeValues = radioisotopeValues[-channelShift:] + radioisotopeValues[:-channelShift]        # Shift the list so that the real radioisotope value is in the real channel
    seedYoungArtifact.attributes["radioisotopeValues"] = radioisotopeValues
    seedYoungArtifact.attributes["radiocarbonAge"] = knownArtifactAges[2]

    # The real channel is the starting channel (0), plus the shift, plus 1 (since channels are numbered 1, 2, 3, 4 in the radioisotope meter instead of 0, 1, 2, 3)
    realChannel = channelShift + 1

    #print("Real channel: " + str(realChannel))
    # TODO: Critical hypotheses
    scoringInfo["criticalHypotheses"] = ["The lower a value an artifact has on Radioisotope Channel " + str(realChannel) + ", the older it is."]

    scoringInfo["criticalQuestions"].append("Does it clearly make the realization (or assumption) that the known artifacts are from the `stone age`, `bronze age`, and `iron age`, or otherwise assume that the stone artifact is the oldest, the bronze artifact is the middle, and the iron artifact is the youngest?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that with radioisotope dating, artifacts that are younger should have higher radioisotope values, and/or older artifacts should have lower values?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that Radioisotope Channel " + str(realChannel) + " shows a pattern where the (1) stone artifact has the lowest value, (2) bronze artifact has a medium value, and (3) iron artifact has the highest value, making it likely useful for radioisotope dating?")

    # Now the 3 unknown artifacts
    unknownArtifacts = []
    unknownArtifactAges = [oldArtifactAge, mediumArtifactAge, youngArtifactAge]
    for i in range(0, 3):
        unknownArtifact = world.createObject("AncientArtifact")
        radioisotopeValues = [realRadioisotopeValues[3+i], fakeRadioisotope1Values[3+i], fakeRadioisotope2Values[3+i], fakeRadioisotope3Values[3+i]]
        radioisotopeValues = radioisotopeValues[-channelShift:] + radioisotopeValues[:-channelShift]        # Shift the list so that the real radioisotope value is in the real channel
        unknownArtifact.attributes["radioisotopeValues"] = radioisotopeValues
        unknownArtifact.attributes["radiocarbonAge"] = unknownArtifactAges[i]
        unknownArtifacts.append(unknownArtifact)

    # Shuffle the order of the unknown artifacts
    world.rng.shuffle(unknownArtifacts)

    # Shuffle the order of the seed artifacts
    seedArtifacts = [seedOldArtifact, seedMediumArtifact, seedYoungArtifact]
    world.rng.shuffle(seedArtifacts)


    # Add dig sites
    scoringInfo["seedArtifacts"] = []
    scoringInfo["unknownArtifacts"] = []
    scoringInfo["signs"] = []
    goldDigSiteIdx = -1
    for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
        if (digSiteIdx < 3):
            # Seed artifact
            artifact = seedArtifacts[digSiteIdx]
            sign = mkDigSiteWithObj(digSiteLocation[0], digSiteLocation[1], world, world.rng, digSiteIdx+1, artifact, artifactExposed=True)
            scoringInfo["seedArtifacts"].append(artifact)
            scoringInfo["signs"].append(sign)
        else:
            # Unknown artifact
            artifact = unknownArtifacts[digSiteIdx-3]
            addedSign = mkDigSiteWithObj(digSiteLocation[0], digSiteLocation[1], world, world.rng, digSiteIdx+1, artifact, artifactExposed=False)
            scoringInfo["unknownArtifacts"].append(artifact)
            scoringInfo["signs"].append(addedSign)
            # Also make a special note of the target (i.e. winning) sign
            if (artifact.attributes["radiocarbonAge"] == oldArtifactAge):
                scoringInfo["targetSign"] = addedSign
                goldDigSiteIdx = digSiteIdx+1

    scoringInfo["criticalHypotheses"].append("The artifact at Dig Site #" + str(goldDigSiteIdx) + " is the oldest, with an age of " + str(oldArtifactAge) + " years. The ages of the other artifacts (in years) are: " + str(mediumArtifactAge) + ", " + str(youngArtifactAge) + ".")

    #scoringInfo["criticalQuestions"].append("Does it clearly state that the artifact  Radioisotope Channel " + str(realChannel) + " value of the artifact at Dig Site #" + str(goldDigSiteIdx) + " is the lowest of the unknown artifacts, signifying that it is the oldest unknown artifact?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that the artifact at Dig Site #" + str(goldDigSiteIdx) + " has the lowest Radioisotope Channel " + str(realChannel) + " value, signifying that it is the oldest of the 3 unknown artifacts?")

    # Add a table at the start of the dig site
    instrumentTable = world.createObject("Table")
    world.addObject(15, 15, Layer.FURNITURE, instrumentTable)
    # Add a radiocarbon meter to the table
    radioisotopeMeter = world.createObject("RadioisotopeMeter")
    instrumentTable.addObject( radioisotopeMeter )
    scoringInfo["radioisotopeMeter"] = radioisotopeMeter

    # Add a shovel at the start of the dig site
    shovel = world.createObject("Shovel")
    world.addObject(14, 15, Layer.FURNITURE, shovel)
    scoringInfo["shovel"] = shovel

    # Add a flag at the start of the dig site
    flag = world.createObject("Flag")
    world.addObject(14, 16, Layer.FURNITURE, flag)
    scoringInfo["flag"] = flag

    # Add some random holes
    minHoles = 1
    holeCount = 0
    while (holeCount < minHoles):
        # Pick a random locatio between 10-20
        randX = world.rng.randint(10, 20)
        randY = world.rng.randint(10, 20)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a hole
                soilTile = world.createObject("SoilTile")
                #soilTile.attributes['hasHole'] = True
                # Remove all soil tile contents (i.e. the dirt) -- this has the effect of making a hole
                for obj in soilTile.contents:
                    soilTile.removeObject(obj)

                world.addObject(randX, randY, Layer.BUILDING, soilTile)
                holeCount += 1


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

    # Add some trees (statically placed)
    mkTallTree(14, 7, world)
    mkTallTree(18, 8, world)
    mkTallTree(8, 12, world)
    mkTallTree(19, 15, world)
    mkTallTree(10, 20, world)
    mkTallTree(20, 19, world)

    # Add some small plants (randomly placed)
    plantCount = 0
    minPlants = 15
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(2, world.sizeX - 2)
        randY = world.rng.randint(2, world.sizeY - 2)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a plant
                world.addObject(randX, randY, Layer.OBJECTS, world.createObject("PlantRandomSmall"))
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


    # Add teleport locations to world
    world.addTeleportLocation("base camp", 13, 14)
    for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
        world.addTeleportLocation("dig site " + str(digSiteIdx+1), digSiteLocation[0]-1, digSiteLocation[1]+1)

    # Return the scoring info
    return scoringInfo


#
#   Archaeology (Easy/Distilled version)
#
def makeScenarioArchaeologicalDigEasyDistilled(world, numUserAgents=1):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []
    numDigSites = 3

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 1
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects
    # Fill with grass
    mkGrassFill(world)
    # Randomly place a few plants (plant1, plant2, plant3)
    for i in range(0, 10):
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)

    # Possible artifact ages (in years). Ranges from 100k to 1 million years old, with even numbers.
    artifactAges = [150000, 230000, 370000, 410000, 500000, 675000, 725000, 890000, 930000, 1000000]
    # Shuffle the artifact ages
    world.rng.shuffle(artifactAges)
    # Trim to the first numDigSites
    artifactAges = artifactAges[:numDigSites]
    # Find the oldest artifact
    oldestArtifactAge = max(artifactAges)

    houseSizeX = 5
    houseSizeY = 5
    mkBuildingOneRoom(world, x=13, y=12, width=houseSizeX, height=houseSizeY, signText="Research Building", doorKeyID=123)  # Door is locked, and no key is provided, to make sure they stay in the building

    # List dig site locations
    digSiteLocations = [(14, 13), (14, 14), (14, 15)]
    # Shuffle the locations
    world.rng.shuffle(digSiteLocations)
    # Add 3 dig sites
    scoringInfo["unknownArtifacts"] = []
    otherArtifactAges = []
    for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
        #artifact, sign = mkDigSite(digSiteLocation[0], digSiteLocation[1], world, world.rng, digSiteIdx+1, artifactAges[digSiteIdx])
        # Make artifact
        artifact = world.createObject("AncientArtifact")
        # Add the age
        artifact.attributes['radiocarbonAge'] = artifactAges[digSiteIdx]
        # Put it on a table
        table = world.createObject("Table")
        world.addObject(digSiteLocation[0], digSiteLocation[1], Layer.FURNITURE, table)
        table.addObject(artifact)

        scoringInfo["unknownArtifacts"].append(artifact)
        #scoringInfo["artifacts"].append(artifact)
        if (artifact.attributes["radiocarbonAge"] == oldestArtifactAge):
            scoringInfo["targetArtifact"] = artifact
        else:
            otherArtifactAges.append(artifactAges[digSiteIdx])

    # TODO: Critical hypotheses
    #scoringInfo["criticalHypotheses"] = ["TODO: Add critical hypotheses here"]

    scoringInfo["criticalHypotheses"].append("The artifact with an age of " + str(oldestArtifactAge) + " years is the oldest. The ages of the other artifacts (in years) are: " + ", ".join([str(age) for age in otherArtifactAges]) + ".")
    scoringInfo["criticalQuestions"].append("Does it clearly state that the artifact with age " + str(oldestArtifactAge) + " years is the oldest?")
    scoringInfo["criticalQuestions"].append("Does it clearly state the ages of the other younger artifacts (in years), which are: `" + ", ".join([str(age) for age in otherArtifactAges]) + "`?")




    # Add a table at the start of the dig site
    instrumentTable = world.createObject("Table")
    world.addObject(16, 13, Layer.FURNITURE, instrumentTable)
    # Add a radiocarbon meter to the table
    radioCarbonMeter = world.createObject("RadioCarbonMeter")
    instrumentTable.addObject( radioCarbonMeter )
    scoringInfo["radioCarbonMeter"] = radioCarbonMeter

    # Add a flag at the start of the dig site
    flag = world.createObject("Flag")
    world.addObject(16, 15, Layer.FURNITURE, flag)
    scoringInfo["flag"] = flag

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


    # Add some trees (statically placed)
    mkTallTree(14, 7, world)
    mkTallTree(18, 8, world)
    mkTallTree(8, 12, world)
    mkTallTree(19, 15, world)
    mkTallTree(10, 20, world)
    mkTallTree(20, 19, world)

    # Add some small plants (randomly placed)
    plantCount = 0
    minPlants = 15
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(2, world.sizeX - 2)
        randY = world.rng.randint(2, world.sizeY - 2)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a plant
                world.addObject(randX, randY, Layer.OBJECTS, world.createObject("PlantRandomSmall"))
                plantCount += 1



    # DialogMaker
    dialogMaker = DialogMaker()

    # Add some number of user agents
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # TODO: Add starting tools for agent
        # Add the agent to a specfic location
        world.addObject(15+userAgentIdx, 14, Layer.AGENT, userAgent)      # Near center of dig site
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)


    # Add teleport locations to world
    world.addTeleportLocation("starting location", 15, 14)
    #for digSiteIdx, digSiteLocation in enumerate(digSiteLocations):
    #    world.addTeleportLocation("dig site " + str(digSiteIdx+1), digSiteLocation[0]-1, digSiteLocation[1]+1)


    # Return the scoring info
    return scoringInfo
