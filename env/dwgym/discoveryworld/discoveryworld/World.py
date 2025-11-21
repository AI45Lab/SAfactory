# World.py

import time
import zlib
import pickle
import json
from os.path import join as pjoin

import pygame

from discoveryworld.SpriteLibrary import SpriteLibrary
from discoveryworld.ObjectMaker import ObjectMaker
from discoveryworld.Layer import Layer
from discoveryworld.TaskScorer import *
from discoveryworld.UUIDGenerator import *
from discoveryworld.DiscoveryFeed import *
from discoveryworld.constants import DATA_PATH

from discoveryworld.JSONEncoder import CustomJSONEncoder

# Storage class for the world (including the full environment grid)
class World:
    # Constructor
    def __init__(self, assetPath, filenameSpriteIndex, dataPath, filenameObjectData, filenameMaterialData, filenameDiscoveryFeed):
        # World size (in tiles)
        self.sizeX = 32
        self.sizeY = 32

        # Default parameters of the world that can change (like background radiation level)
        self.parameters = {
            "avgRadiationBackgroundLevelUSVH": 0.05        # Average background radiation level in uSV/h
        }

        # UUID Generator
        self.uuidGenerator = UUIDGenerator()

        # Load sprites
        self.spriteLibrary = SpriteLibrary(assetPath, filenameSpriteIndex)

        # Load object data
        self.objectMaker = ObjectMaker(dataPath, filenameObjectData, filenameMaterialData, world=self, knownSpriteNames=self.spriteLibrary.getSpriteNames())

        # Initialize grid
        self.grid = [[self.mkBlankGridTile() for x in range(self.sizeX)] for y in range(self.sizeY)]

        # Initialize agent array
        self.agents = []

        # Initialize DiscoveryFeed, and load any pre-existing articles and update posts
        self.discoveryFeed = DiscoveryFeed(pjoin(dataPath or DATA_PATH, filenameDiscoveryFeed))

        # Initialize task generator and scorer
        self.taskMaker = TaskMaker(world=self)
        self.taskScorer = TaskScorer(world=self)

        # Load world data
        # TODO

        # Step number (i.e. number of ticks that have passed)
        self.step = 0

        # Font
        self.font = pygame.font.SysFont("Arial", 8)

        # World history
        self.worldHistory = []

        # Add a dictionary of teleport locations (key = location name), to make an agent's navigation task easier
        self.teleportLocations = {}

        # Add start time
        self.startTime = time.time()

        # Add a flag for whether this has a live user playing (where text is somewhat simplified, or an agent)
        self.liveUserPlaying = False

        # These are set in the ScenarioMaker
        self.randomSeed = None
        self.rng = None


    # Set if a live user is playing (e.g. for a user study)
    def setLiveUserPlaying(self):
        self.liveUserPlaying = True

    #
    #   Initialization (agents, tasks)
    #

    # Add an agent to the world
    def addAgent(self, agent):
        self.agents.append(agent)

    # Return an agent by name
    def getAgentByName(self, agentName):
        for agent in self.agents:
            if agent.name == agentName:
                return agent

        return None

    # Return a list of all agents that are user agents
    def getUserAgents(self):
        userAgents = []
        for agent in self.agents:
            if (agent.attributes["isNPC"] == False):
                userAgents.append(agent)
        return userAgents

    # Return a list of all agents that are NPC agents
    def getNPCAgents(self):
        npcAgents = []
        for agent in self.agents:
            if (agent.attributes["isNPC"] == True):
                npcAgents.append(agent)
        return npcAgents

    def addTaskByName(self, taskName, scoringInfo):
        task = self.taskMaker.makeTask(taskName, scoringInfo)
        if task != None:
            self.taskScorer.addTask(task)
            task.taskSetup()
            return True
        else:
            print("Error: Task name not recognized: " + taskName)
            return False



    #
    # Get the current step number.  Starts at 0, increments 1 per call to the world.tick().
    #
    def getStepCounter(self):
        return self.step


    #
    #   Grid
    #

    # Create a blank grid tile
    def mkBlankGridTile(self):
        gridTile = {
            "layers": {}
        }

        for layer in Layer:
            gridTile["layers"][layer] = []

        return gridTile


    # Add an object to the world
    def addObject(self, x, y, layer, object:Object):
        # Bound checking: Make sure the object is within the world bounds
        #if x < 0 or x >= self.sizeX or y < 0 or y >= self.sizeY:
        if (self.isWithinBounds(x, y) == False):
            print("Error: Object out of bounds: " + str(x) + ", " + str(y))
            return False


        # Remove the object from its current container
        object.removeSelfFromContainer()

        ## TODO: Does it also have to be removed from the current grid location on the world grid? (e.g. 'addObject'ing the same object to two locations currently appears to render duplicate copies of the same object)

        # Set the object's position
        object.setWorldLocation(x, y)

        # World, building, and furniture layers are overwrite layers (i.e. they can hold only a single object)
        # if layer == Layer.WORLD or layer == Layer.BUILDING or layer == Layer.FURNITURE:
        #     self.grid[x][y]["layers"][layer] = []
        #     self.grid[x][y]["layers"][layer].append(object)
        # # Objects layer is an additive layer (i.e. it can hold multiple objects)
        # elif layer == Layer.OBJECTS:
        #     self.grid[x][y]["layers"][layer].append(object)
        # # Player layer is a special layer that can only hold a single object
        # elif layer == Layer.AGENT:
        #     self.grid[x][y]["layers"][layer] = []
        #     self.grid[x][y]["layers"][layer].append(object)

        # All layers can hold multiple objects (prevents objects suddenly disappearing when a new object is moved to the same world location)
        if layer in Layer:
           self.grid[x][y]["layers"][layer].append(object)
        else:
            raise ValueError("Error: Invalid layer: " + str(layer))

    # Get all objects at a given position
    def getObjectsAt(self, x, y, respectContainerStatus=False, includeParts=False, excludeObjectsOnAgents=False, respectObscuringLowerLayers=False, includeContents=True):
        # Bound checking: Make sure the object is within the world bounds
        if x < 0 or x >= self.sizeX or y < 0 or y >= self.sizeY:
            #print("Error: Object out of bounds: " + str(x) + ", " + str(y))
            return []

        #print("\t\t\tGetting objects at (" + str(x) + ", " + str(y) + ")")
        # Get the objects
        objects = []
        #for layer in Layer:
        for layer in reversed(list(Layer)):         # Reverse the order of layers, so that the top layer is processed first, to handle obscuring objects
            #objects += self.grid[x][y]["layers"][layer]
            objsToAdd = self.grid[x][y]["layers"][layer]
            if (len(objsToAdd) > 0):
                #print("\t\t\t\tLayer: " + str(layer) + " (" + str(len(objsToAdd)) + " objects)")
                foundObscuringObject = False

                for obj in objsToAdd:
                    objects.append(obj)

                    # If this object obscures objects below it, then stop adding objects to the list
                    if (obj.attributes["obscuresObjectsBelow"] == True):
                        foundObscuringObject = True

                    contents = []
                    # Check if this object is an agent (i.e. is an instance of Agent)
                    if (excludeObjectsOnAgents == True) and (obj.attributes["isAgent"] == True):
                        #print("\t\t\t\t\tSkipping agent " + obj.name)
                        continue

                    # Add contents/parts
                    if (includeParts == False) and (includeContents == True):
                        contents = obj.getAllContainedObjectsRecursive(respectContainerStatus=respectContainerStatus)
                    elif (includeParts == True) and (includeContents == False):
                        contents = obj.getAllContainedObjectsAndParts(includeContents=False, includeParts=True)
                    elif (includeParts == True) and (includeContents == True):
                        contents = obj.getAllContainedObjectsAndParts(includeContents=True, includeParts=True)
                    if (len(contents) > 0):
                        #print("Contents of " + obj.name + ": " + str(contents))
                        for cObj in contents:
                            if (cObj.attributes["obscuresObjectsBelow"] == True):
                                foundObscuringObject = True
                            objects.append(cObj)

                # If we found an object that obscures objects below it, then stop adding objects to the list
                if ((respectObscuringLowerLayers == True) and (foundObscuringObject == True)):
                    break

            #objects += objsToAdd


            #for obj in self.grid[x][y]["layers"][layer]:
            #    objects += obj.getAllContainedObjectsRecursive(respectContainerStatus=True)

        return objects

    # Helper to get all world objects (for example, for scoring)
    def getAllWorldObjects(self):
        allObjects = []
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                allObjects += self.getObjectsAt(x, y)
        return allObjects

    # Helper to get a specific world object, by its UUID
    def getObjectByUUID(self, uuid):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                allObjectsThisTile = self.getObjectsAt(x, y)
                for obj in allObjectsThisTile:
                    if obj.uuid == uuid:
                        return obj
        return None


    # Remove an object from the world
    def removeObject(self, object:Object):
        # First, get the object's position (stored internally in that object)
        objX, objY = object.getWorldLocation()
        # Then, set the object's position to -1 (i.e. not in the world)
        object.setWorldLocation(-1, -1)

        # Remove the object from its current container
        object.removeSelfFromContainer()

        # Remove the object from the world
        if (objX >= 0) and (objY >= 0):
            for layer in Layer:
                if object in self.grid[objX][objY]["layers"][layer]:
                    # Remove object from this layer
                    self.grid[objX][objY]["layers"][layer].remove(object)
                    # Success
                    return True



        # If we reach here, something went wrong -- the object could not be removed from the world.
        #print("WARNING: Object could not be removed from the world (" + object.name + ") at (" + str(objX) + ", " + str(objY) + ")")
        return True



    #
    #   Object Generation (light wrapper around ObjectMaker)
    #

    def createObject(self, objectReferenceName, *args, **kwargs):
        return self.objectMaker.createObject(objectReferenceName, *args, **kwargs)


    #
    #   Helpers for quickly checking world contents
    #

    # Returns true if the given location has a non-zero number of objects in layers: building, furniture, objects, agent
    def doesTileHaveObjectsOnIt(self, x, y):
        checkLayers = [Layer.BUILDING, Layer.FURNITURE, Layer.OBJECTS, Layer.AGENT]

        # Bound checking: Make sure the object is within the world bounds
        if x < 0 or x >= self.sizeX or y < 0 or y >= self.sizeY:
            print("Error: Object out of bounds: " + str(x) + ", " + str(y))
            return False

        # Check if the tile has any objects
        for layer in checkLayers:
            if len(self.grid[x][y]["layers"][layer]) > 0:
                return True

        # If we reach here, the tile has no objects
        return False


    #
    #   Traversability
    #

    # Check if a tile is passable
    # Returns (bool, obj) where:
    #  - bool is true if the tile is passable, and obj is the first impassable object encountered at that location.
    #  - If the tile is passable, obj is None.
    def isPassable(self, x, y):
        # Bound checking: Make sure the object is within the world bounds
        if x < 0 or x >= self.sizeX or y < 0 or y >= self.sizeY:
            #print("Error: Object out of bounds: " + str(x) + ", " + str(y))
            return (False, None)

        # Check if the tile is passable
        # Collect all the objects across all layers
        allObjs = self.getObjectsAt(x, y)
        # Check if any of the objects are impassable
        for object in allObjs:
            if (not object.attributes["isPassable"]):
                return (False, object)

        # If we reach here, the tile is passable
        return (True, None)


    #
    #   Bound Checking
    #

    # Returns true if the specified position is within the world grid bounds
    def isWithinBounds(self, x, y):
        return (x >= 0) and (x < self.sizeX) and (y >= 0) and (y < self.sizeY)

    #
    #   World update
    #
    def tick(self):
        # Update all objects in the world

        # First, reset whether each object has had its tick() function called this tick
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for layer in Layer:
                    for object in self.grid[x][y]["layers"][layer]:
                        object.resetTick()

                # alObjs = self.getObjectsAt(x, y, respectContainerStatus=False, includeParts=True)   # We need to do it this way, to make sure we reset the tick for all the nested parts.
                # for object in alObjs:
                #         object.tickCompleted = False


        # Then, call tick() on each object in the world
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for layer in Layer:
                    for object in self.grid[x][y]["layers"][layer]:
                        object.tick()


        # Also do a tick of the task scorer, to measure task progress
        self.taskScorer.updateTick()

        # Also save the world history
        self.saveWorldHistory()

        # Collect any signals from posts sent in the current Discovery Feed step
        signalsToBroadcast = self.discoveryFeed.getSignalsFromPosts(curStep = self.step)

        # Broadcast those signals to all agents
        for agent in self.agents:
            for signal in signalsToBroadcast:
                print("BROADCASTING SIGNAL " + str(signal) + " to agent " + agent.name)
                agent.addState(signal)

        # Increment step counter
        self.step += 1


    #
    #   Saving a world history
    #
    def saveWorldHistory(self):
        print("Saving world history...")

        # Keep track of time
        startTime = time.time()

        # Task scoring
        taskScores = []
        for task in self.taskScorer.tasks:
            taskScores.append(task.taskProgressDict())


        # First, create a new world history dictionary.
        packed = {
            "step": self.step,
            "sizeX": self.sizeX,
            "sizeY": self.sizeY,
            "grid": [],
            "discoveryFeed": self.discoveryFeed.toDict(),
            "taskScores": taskScores,
            "runtime_seconds": round(time.time() - self.startTime, 1),
        }

        # Clone everything in the grid into this record
        totalObjects = 0
        for x in range(self.sizeX):
            packed["grid"].append([])
            for y in range(self.sizeY):
                # Add all objects in this tile
                allObjs = []
                for layer in Layer:
                    for object in self.grid[x][y]["layers"][layer]:

                        # For each object, get all the objects it contains/that are its parts.
                        allObjContentsAndParts = object.getAllContainedObjectsAndParts()
                        # Add them to the list of objects to save
                        for obj in allObjContentsAndParts:
                            # Pack to dict using the to_dict() method
                            allObjs.append( obj.to_dict() )
                            totalObjects += 1

                packed["grid"][x].append(allObjs)

        # Save the history
        #packed["objects"] = allObjs

        # Compress the world history using zlib and pickle (saves almost 99% of space!)
        packedCompressed = zlib.compress(pickle.dumps(packed))

        self.worldHistory.append(packedCompressed)

        print("Number of items in world history:" + str(len(self.worldHistory)))
        print("Size of most recent item: " + str(len(packedCompressed)) + " bytes " + str(len(packedCompressed) / 1024) + " KB")

        # delta time
        deltaTime = time.time() - startTime
        print("World history includes " + str(totalObjects) + " objects.")
        print("World history saved in " + str(deltaTime) + " seconds.")


    def getWorldHistoryAtStep(self, step):
        # Return the world history at the specified step
        # Since the world history is pickled then compressed, this will require uncompressing then unpickling
        if (step < 0 or step >= len(self.worldHistory)):
            return None

        # Get compressed history for a given step
        compressedPickled = self.worldHistory[step]

        # Uncompress the history
        pickled = zlib.decompress(compressedPickled)
        # Unpickle the history
        history = pickle.loads(pickled)

        return history

    # Export the world history to a (pickled)
    def exportWorldHistory(self, filename):
        print("Exporting world history to file: " + filename + "...")
        f = open(filename, "wb")
        f.write(pickle.dumps(self.worldHistory))
        f.close()
        print("Export complete.")

    # Export the world history to JSON
    def exportWorldHistoryJSON(self, logInfo, filename, pygameWindow, pygame, lastScreenExportFilename):
        print("Exporting world history to JSON file: " + filename + "...")

        # Break apart into multiple files, each containing a maximum of 100 steps
        # This is necessary because the JSON file can get very large

        # First, figure out how many "parts" we need to break the history into
        MAX_STEPS_PER_FILE = 100
        numParts = int(len(self.worldHistory) / MAX_STEPS_PER_FILE) + 1
        originalFilename = filename
        for partIdx in range(numParts):
            startStep = partIdx * MAX_STEPS_PER_FILE
            endStep = (partIdx + 1) * MAX_STEPS_PER_FILE

            filenamePart = originalFilename.replace(".json", ".part" + str(partIdx) + "of" + str(numParts) + ".json")
            print("Exporting part " + str(partIdx) + " of " + str(numParts) + " to file: " + filenamePart + "...")

            # Deep copy logInfo
            logInfoCopy = logInfo.copy()
            logInfoCopy["partIdx"] = partIdx
            logInfoCopy["numParts"] = numParts
            logInfoCopy["startStep"] = startStep
            logInfoCopy["endStep"] = endStep

            with open(filenamePart, 'w') as outfile:
                # First, export the logInfo
                outfile.write("{\n")
                outfile.write("\t\"logInfo\": " + json.dumps(logInfoCopy) + ",\n")
                outfile.write("\t\"worldHistory\": [\n")
                #json.dump(packed, outfile, indent=4)
                # Actually must export each step in the world history separately, since it has to be unpacked
                #for step in range(len(self.worldHistory)):
                endRange = min(endStep, len(self.worldHistory))
                for step in range(startStep, endRange):
                    if (step % 50 == 0):
                        print("Exporting step " + str(step) + " of " + str(len(self.worldHistory)) + "...")

                    historyStep = self.getWorldHistoryAtStep(step)
                    #json.dump(history, outfile, indent=4)
                    json.dump(historyStep, outfile, indent=2, cls=CustomJSONEncoder)

                    outfile.write("\n")
                    if (step < endRange - 1):
                        outfile.write(",\n")

                    # Show a progress indicator on the PyGame window
                    if (pygameWindow != None) and (pygame != None):
                        pygameWindow.fill((0, 0, 0))
                        # Display the last screenshot on the window as a background
                        if (lastScreenExportFilename != None):
                            #img = pygame.image.load(lastScreenExportFilename)
                            #pygameWindow.blit(img, (0, 0))
                            # As above, but make it faded out
                            img = pygame.image.load(lastScreenExportFilename)
                            img.set_alpha(128)
                            pygameWindow.blit(img, (0, 0))

                        # Blank a rectangle in the middle of the screen
                        rectWidth = 400
                        rectHeight = 80
                        rectX = (pygameWindow.get_width() - rectWidth) / 2
                        rectY = (pygameWindow.get_height() - rectHeight) / 2
                        pygame.draw.rect(pygameWindow, (100, 100, 100), (rectX, rectY, rectWidth, rectHeight))

                        font = pygame.font.SysFont("monospace", 15, bold=True)
                        text1 = font.render("Saving History (this may take a moment...)", True, (200, 200, 200))
                        text2 = font.render(str(step) + " / " + str(len(self.worldHistory)), True, (200, 200, 200))

                        # Center text in the rectangle
                        textX = rectX + (rectWidth - text1.get_width()) / 2
                        textY = rectY + (rectHeight - text1.get_height()) / 2
                        pygameWindow.blit(text1, (textX, textY - 20))
                        textX = rectX + (rectWidth - text2.get_width()) / 2
                        pygameWindow.blit(text2, (textX, textY + 20))

                        pygame.display.flip()

                outfile.write("\t]\n")
                outfile.write("}\n")

            # Compress the file into a ZIP
            import zipfile
            zipFilename = filenamePart.replace(".json", ".zip")
            print("Compressing part " + str(partIdx) + " of " + str(numParts) + " to file: " + zipFilename + "...")
            with zipfile.ZipFile(zipFilename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(filenamePart, arcname=filenamePart.split("/")[-1])
            # Remove the original JSON file
            import os
            os.remove(filenamePart)


        print("Export complete.")

    #
    #   Rendering
    #
    def render(self, window, cameraX, cameraY):
        #world.spriteLibrary.displaySprite("house1_house_corner_tl", window, 0, 0)
        # Render the world
        for y in range(self.sizeY):
            for x in range(self.sizeX):

                # Render the world layer
                for object in self.grid[x][y]["layers"][Layer.WORLD]:
                    #print("Rendering: " + object.name)
                    #self.spriteLibrary.renderSprite(window, object.getSpriteName(), x * 32 - cameraX, y * 32 - cameraY)
                    #for spriteName in object.getSpriteNamesWithContents():
                    #    self.spriteLibrary.renderSprite(window, spriteName, x * 32 - cameraX, y * 32 - cameraY)
                    for spriteDict in object.getSpriteNamesWithContents():
                        self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], x * 32 - cameraX, y * 32 - cameraY + spriteDict["yOffset"])

                # Render the building layer
                for object in self.grid[x][y]["layers"][Layer.BUILDING]:
                    #print("Rendering: " + object.name)
                    #self.spriteLibrary.renderSprite(window, object.getSpriteName(), x * 32 - cameraX, y * 32 - cameraY)
                    #for spriteName in object.getSpriteNamesWithContents():
                    #    self.spriteLibrary.renderSprite(window, spriteName, x * 32 - cameraX, y * 32 - cameraY)
                    for spriteDict in object.getSpriteNamesWithContents():
                        self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], x * 32 - cameraX, y * 32 - cameraY + spriteDict["yOffset"])


                # Render the furniture layer
                for object in self.grid[x][y]["layers"][Layer.FURNITURE]:
                    #print("Rendering: " + object.name)
                    #self.spriteLibrary.renderSprite(window, object.getSpriteName(), x * 32 - cameraX, y * 32 - cameraY)
                    #for spriteName in object.getSpriteNamesWithContents():
                    #    self.spriteLibrary.renderSprite(window, spriteName, x * 32 - cameraX, y * 32 - cameraY)
                    for spriteDict in object.getSpriteNamesWithContents():
                        self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], x * 32 - cameraX, y * 32 - cameraY + spriteDict["yOffset"])

                # Render the objects layer
                for object in self.grid[x][y]["layers"][Layer.OBJECTS]:
                    #print("Rendering: " + object.name)
                    #self.spriteLibrary.renderSprite(window, object.getSpriteName(), x * 32 - cameraX, y * 32 - cameraY)
                    #for spriteName in object.getSpriteNamesWithContents():
                    #    self.spriteLibrary.renderSprite(window, spriteName, x * 32 - cameraX, y * 32 - cameraY)
                    for spriteDict in object.getSpriteNamesWithContents():
                        self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], x * 32 - cameraX, y * 32 - cameraY + spriteDict["yOffset"])


                # Render the player layer
                for object in self.grid[x][y]["layers"][Layer.AGENT]:
                    #print("Rendering: " + object.name)
                    #self.spriteLibrary.renderSprite(window, object.getSpriteName(), x * 32 - cameraX, y * 32 - cameraY)
                    #for spriteName in object.getSpriteNamesWithContents():
                    #    self.spriteLibrary.renderSprite(window, spriteName, x * 32 - cameraX, y * 32 - cameraY)
                    for spriteDict in object.getSpriteNamesWithContents():
                        self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], x * 32 - cameraX, y * 32 - cameraY + spriteDict["yOffset"])



    #   Render a viewport, starting from world position (worldStartX, worldStartY), and of size (in tiles) (sizeTilesX, sizeTilesY).
    #   The offset (in pixels) from the top-left corner of the window is (offsetX, offsetY)
    def renderViewport(self, window, worldStartX, worldStartY, sizeTilesX, sizeTilesY, offsetX, offsetY, scale=1.0, includeGrid=False):
        # Render the viewport
        originalTileSize = int(32)
        tileSize = int(32 * scale)

        # Handle the wonkyness of the scaling as it relates to slight Y drawing offsets (particularly for non-standard (i.e. non-32-pixel) sprites)
        tileDiff = tileSize - originalTileSize
        #offsetY -= tileDiff

        # DEBUG: Enable rendering grid locations
        renderGridLocations = includeGrid
        # For agents: Enable rendering grid (expands the tile size by one, to leave a functional black grid line between tiles)
        # if (includeGrid):
        #    tileSize += 1

        # Render the world layers in order.
        for layer in [Layer.WORLD, Layer.BUILDING, Layer.FURNITURE, Layer.OBJECTS, Layer.AGENT, Layer.AIR]:

            for y in range(worldStartY, worldStartY + sizeTilesY):
                for x in range(worldStartX, worldStartX + sizeTilesX):
                    # Bound checking: Make sure the object is within the world bounds
                    if (x < 0) or (x >= self.sizeX) or (y < 0) or (y >= self.sizeY):
                        #print("Error: Object out of bounds: " + str(x) + ", " + str(y))
                        continue

                    # Determine screen x/y coordinates
                    screenX = (x - worldStartX) * tileSize + offsetX
                    screenY = (y - worldStartY) * tileSize + offsetY

                    for object in self.grid[x][y]["layers"][layer]:
                        #object.render(self.spriteLibrary, window, screenX, screenY, scale)
                        for spriteDict in object.getSpriteNamesWithContents():
                            self.spriteLibrary.renderSprite(window, spriteDict["spriteName"], screenX + spriteDict.get("xOffset", 0), screenY + spriteDict.get("yOffset", 0), scale * spriteDict.get("scale", 1.0))


        if renderGridLocations or includeGrid:
            for y in range(worldStartY, worldStartY + sizeTilesY):
                for x in range(worldStartX, worldStartX + sizeTilesX):
                    # Bound checking: Make sure the object is within the world bounds
                    if (x < 0) or (x >= self.sizeX) or (y < 0) or (y >= self.sizeY):
                        #print("Error: Object out of bounds: " + str(x) + ", " + str(y))
                        continue

                    # Determine screen x/y coordinates
                    screenX = (x - worldStartX) * tileSize + offsetX
                    screenY = (y - worldStartY) * tileSize + offsetY

                    # Draw a rectangle around the tile
                    pygame.draw.rect(window, (0, 0, 0), (screenX, screenY, tileSize, tileSize), 1)

                    # If enabled, render the grid location (for debugging)
                    if (renderGridLocations):
                        text = self.font.render(str(x) + "," + str(y), True, (0, 0, 0))
                        window.blit(text, (screenX, screenY))

    #
    #   Teleport locations
    #
    def addTeleportLocation(self, name, gridX, gridY):
        self.teleportLocations[name] = {"gridX": gridX, "gridY": gridY}

    def getTeleportLocations(self):
        return self.teleportLocations


    def hasObj(self, x, y, type):
        objects = self.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if (object.type == type):
                return True
        return False


    #
    #   Filtering
    #

    # Filter out objects that can never be visible -- e.g. initial Grass tiles that are always covered by buildings
    def initialFilter(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                # Get all the objects at this location
                objects = self.getObjectsAt(x, y)

                grassObjs = [x for x in objects if (x.type == "grass")]
                obscuringObjects = []
                obscuringTypes = ["path", "floor", "cave floor", "soil", "cave wall"]
                foundObscuringObject = False
                for obj in objects:
                    if (obj.type in obscuringTypes):
                        foundObscuringObject = True
                        break
                if (foundObscuringObject):
                    for obj in grassObjs:
                        self.removeObject(obj)
                        #print("Removing grass object at (" + str(x) + ", " + str(y) + ")")


    #
    #   Pathfinding
    #

    # Get the next step in the path from one location to another
    #def getNextStep(self, fromLocation, toLocation):
