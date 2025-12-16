# ObjectModel.py

import math
import random

import numpy as np

from discoveryworld import ActionSuccess


# Storage class for a single object
class Object:

    # Constructor
    def __init__(self, world, objectType, objectName, defaultSpriteName, rngSeed=None):
        self.type = objectType
        self.name = objectName
        self.defaultSpriteName = defaultSpriteName
        self.uuid = world.uuidGenerator.generateUUID()          # Generate a unique integer to represent this object
        self.rng = random.Random()                              # Random number generator for this object

        # Whether the agent has had tick() called already this past update
        self.tickCompleted = False

        # Whether the sprite name needs to be recalculated
        # By default this is off (i.e. static sprites).
        # If your object changes sprites based on status, then you should set this to True,
        # especially in tick() whenever the object changes state.
        self.needsSpriteNameUpdate = False

        # Name of the current and last sprite names
        self.curSpriteName = None
        self.curSpriteModifiers = set()
        self.lastSpriteName = None
        self.tempLastSpriteName = None      # Used to store what the next 'lastSpriteName' will be -- updated when the backbuffer flips

        # World back-reference
        self.world = world

        # Properties/attributes
        self.attributes = {}

        # Initial world location (undefined)
        if ("gridX" not in self.attributes):
            self.attributes["gridX"] = -1
        if ("gridY" not in self.attributes):
            self.attributes["gridY"] = -1

        # Agent action history (if applicable)
        self.actionHistory = None
        self.attributes["isAgent"] = False                          # Is this object an agent? (e.g. a person)
        self.attributes["isNPC"] = False                            # Is this agent an NPC?

        # Default attributes
        self.attributes["isMovable"] = True                         # Can it be moved?
        self.attributes["isPassable"] = True                        # Can an agent walk over this?

        # Whether this object obscures objects on lower layers (like a floor tile on the furniture layer obscuring grass or soil on the world layer)
        self.attributes["obscuresObjectsBelow"] = False            # Does it obscure/hide objects on layers below it?

        # Rendering attributes
        self.attributes["screenXOffset"] = 0                        # Small X offset in rendering contents. This is to make it look like (e.g.) the contents of an object (like a table) are sitting on it.
        self.attributes["screenYOffset"] = 0                        # Small Y offset in rendering contents. This is to make it look like (e.g.) the contents of an object (like a table) are sitting on it.

        # Materials
        self.attributes["materials"] = []                           # List of materials that this object is made of
        self.attributes["manualMaterialNames"] = []                 # A list of material types to add during initialization (in code, rather than from the spreadsheet)

        # Contents (for containers)
        self.parentContainer = None                                 # Back-reference for the container that this object is in
        self.attributes['isContainer'] = False                      # Is it a container?
        self.attributes['isOpenable'] = False                       # If it's a container, can you open/close it?
        self.attributes['isOpenContainer'] = False                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = ""                     # Container prefix (e.g. "in" or "on")
        self.attributes['isOpen'] = False                           # Closed by default
        self.attributes['contentsVisible2D'] = True                 # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)
        self.contents = []                                          # Contents of the container (other objects)

        # Parts (for composite objects -- similar to containers)
        self.parts = []                                             # List of parts that this object is made of

        # Whether the object has a hole in it (e.g. a hole in the ground), that allows objects to be contained inside it.
        self.attributes['hasHole'] = False                          # Does it have a hole?

        # Passage (for dynamic passages like doors, that can be opened/closed)
        self.attributes['isPassage'] = False                        # Is this a passage?

        # Device (is activable)
        self.attributes['isActivatable'] = False                    # Is this a device? (more specifically, can it be activated/deactivated?)
        self.attributes['isActivated'] = False                      # Is this device currently activated?
        self.attributes['isUsable'] = False                         # Can this device be used with another object? (e.g. specifically through the 'use' action)

        # Dialog attributes
        self.attributes['isDialogable'] = False                     # Can it be dialoged with?

        # Food attributes
        self.attributes['isEdible'] = False                         # Can it be eaten?
        self.attributes['isCooked'] = False                         # Is it cooked?

        # Poison/health attributes
        self.attributes['isPoisonous'] = False                     # Is it poisonous?

        # Can it be used with a shovel (e.g. dug up?).  The function is called useWithShovelResult().
        self.attributes['isShovelable'] = False                    # Can it be shoveled?

        # Readable
        self.attributes['isReadable'] = False                      # Can it be read?
        self.attributes['document'] = ""                           # Any text to read

        # Temperature
        self.attributes['temperatureC'] = 20                       # The default object temeprature, in Celsius
        self.attributes['heatSourceMaxTemp'] = 0                   # If it is a heat source, then this is the maximum temperature that it can reach
        self.attributes['coolSourceMinTemp'] = 0                   # If it is a cool source, then this is the minimum temperature that it can reach

        # Alive
        self.attributes['isLiving'] = False                         # Is it alive?

        # Modifier text, if the object is viewed under a microscope. This is a list of strings, which are displayed in the microscope view.
        self.attributes['microscopeModifierText'] = []

        # Substance properties
        self.attributes["substanceName"] = ""                     # Name of the substance
        self.attributes['isSubstance'] = False                    # Is it a substance?
        self.attributes['isAutoReacting'] = False                 # Does it react automatically with other substances?
        self.attributes['mixtureDict'] = {}                       # Dictionary of substances and their proportions in the mixture

        # Keys
        self.attributes['requiresKey'] = 0                        # If it requires a key to open/use, then this is a special ID for the key.  If the value is <=0, then it doesn't require a key.
        self.attributes['keyID'] = 0                              # If this object acts as a key, here's it's ID (0 by default)

        # Radiocarbon dating age (in years)
        self.attributes["radiocarbonAge"] = -1                    # Radiocarbon dating age (in years). -1 means it's not applicable/inconclusive.
        self.attributes["radioisotopeValues"] = []                # Radioisotope values.  If empty, then it's not applicable/inconclusive.

        # Soil attributes
        self.attributes["soilNutrients"] = {}                    # Soil nutrients.  If empty, then it's not applicable/inconclusive.
        self.attributes["needsNutrientLevels"] = {}              # For seeds/plants: What nutrient levels do they need to grow?
        self.attributes["antirequirementsNutrientLevels"] = []   # A list of dictionaries, each containing a list of nutrient levels under which the seed/plant will NOT grow

        # Object density
        self.attributes["density"] = 0.0                         # Object density (in g/cm^3). <=0 means it's not applicable/inconclusive.

        # Force a first infer-sprite-name
        # NOTE: Moved to a global update (since other objects that the sprite depends on may not be populated yet when it is created)
        self.firstInit = True

    def seed(self, seed):
        # Seed the random number generator
        self.rng.seed(seed)

    #
    # World location
    #

    # This should always be called whenever the object is moved, to ensure that the world location is always up-to-date
    def setWorldLocation(self, gridX, gridY):
        self.attributes["gridX"] = gridX
        self.attributes["gridY"] = gridY

        # Recursively update the world location of all contained objects and parts
        for obj in self.contents:
            obj.setWorldLocation(gridX, gridY)

        for obj in self.parts:
            obj.setWorldLocation(gridX, gridY)

    def getWorldLocation(self):
        # Get the world location of the object
        return (self.attributes["gridX"], self.attributes["gridY"])

    # (x0, y0) should be the top-left corner of the object, and (x1, y1) should be the bottom-right corner
    def isWithinLocationBounds(self, x0, y0, x1, y1):
        # Swap the bounds if they're not in the right order
        if (x0 > x1):
            x0, x1 = x1, x0
        if (y0 > y1):
            y0, y1 = y1, y0
        # Check to see if the object is within the given bounds
        x, y = self.getWorldLocation()
        if (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
            return True

        return False

    # Get the distance from this object to another object
    def distanceTo(self, otherObj):
        # Get the distance from this object to another object
        x0, y0 = self.getWorldLocation()
        x1, y1 = otherObj.getWorldLocation()
        return math.sqrt( (x1 - x0)**2 + (y1 - y0)**2 )

    # Remove from world location -- this is analagous to the removeObject() method for containers, but removes the object from the world tile.
    def removeFromWorldLocation(self):
        # TODO: Depricated? use self.world.removeObject(obj) instead?

        # Remove this object from its world location
        #self.world.removeObjectFromTile(self)
        #self.world.removeObject(self)
        # Remove it from the world (if the world location is positive, signifying it hasn't already been removed)
        if (self.attributes["gridX"] >= 0) and (self.attributes["gridY"] >= 0):
            self.world.removeObject(self)


    #
    #   Material Property Initialization
    #
    def initializeMaterialProperties(self, materialIndexDict):
        # If the object has special material requirements (like randomly choosing from a number of different materials), then this function should be overridden.
        # Otherwise, it can be left blank.

        # Check whether this object has an attribute called 'manualMaterialNames'
        if ("manualMaterialNames" in self.attributes):
            for materialName in self.attributes["manualMaterialNames"]:
                # Add the material to the object
                import copy
                if (materialName not in materialIndexDict):
                    print("ERROR: initializeMaterialProperties: materialName not found in materialIndexDict: " + materialName)
                else:
                    material = copy.deepcopy(materialIndexDict[materialName])
                    self.attributes['materials'].append(material)


    #
    #   Container Semantics
    #

    # Add an object (obj) to this container
    def addObject(self, obj, force=False):
        # Remove the object from its previous container
        obj.removeSelfFromContainer()
        # Remove from the old world location
        self.world.removeObject(obj)
        # Add an object to this container
        self.contents.append(obj)
        # Set the parent container
        obj.parentContainer = self
        # Set the world location to the same as the parent container
        obj.setWorldLocation(self.attributes["gridX"], self.attributes["gridY"])


    # Add an object (obj) as a part of this object
    def addPart(self, obj):
        # Remove the object from its previous container
        obj.removeSelfFromContainer()

        # Add an object as part (using parentContainer as a backreference to whole)
        self.parts.append(obj)
        obj.parentContainer = self

    # Remove an object from this container
    # TODO: Should also remove it from specific world coordinates?
    def removeObject(self, obj):
        #self.removeFromWorldLocation()
        # Remove an object from this container
        if (obj in self.contents):
            self.contents.remove(obj)
            obj.parentContainer = None
            return True
        elif (obj in self.parts):
            self.parts.remove(obj)
            obj.parentContainer = None
            return True
        else:
            return False

    # Remove the current object from whatever container it's currently in
    def removeSelfFromContainer(self):
        # Remove it from the world, too
        #self.removeFromWorldLocation()
        # Remove this object from its container
        if (self.parentContainer != None):
            return self.parentContainer.removeObject(self)


    # Replace self with another object (obj) in the container that it's currently in
    # Largely intended for objects that change into completely different objects.
    def replaceSelfWithObject(self, obj):
        # Replace self with another object in the container that it's currently in
        if (self.parentContainer != None):
            # Get the index of self in the parent container
            idx = self.parentContainer.contents.index(self)
            parentContainerCopy = self.parentContainer
            # Remove self from the parent container
            self.parentContainer.removeObject(self)
            # Add the new object at the same index
            parentContainerCopy.contents.insert(idx, obj)
            # Set the parent container
            obj.parentContainer = parentContainerCopy


    # Get all contained objects
    def getAllContainedObjectsRecursive(self, respectContainerStatus=False):
        # Get all contained objects, recursively
        out = []
        ##print("getAllContainedObjectsRecursive: " + self.name + " (" + str(self.attributes['isOpenContainer']) + ") ." )
        ##print("\tList of all objects in this container: " + str([obj.name for obj in self.contents]))
        # If this is a container, and it's open, then add the contents
        if (not respectContainerStatus) or (respectContainerStatus and self.attributes['isOpenContainer']):
            for obj in self.contents:

                ## DEBUG
                # if (obj.parentContainer != None):
                #     print("\tFound " + obj.name + " (" + str(obj.uuid) + "), which is contained in: " + obj.parentContainer.name + " (" + str(obj.parentContainer.uuid) + ")")
                #     if (len(obj.contents) > 0):
                #         print("\t\t it has the following immediate contents: ")
                #         for obj2 in obj.contents:
                #             print("\t\t\t" + obj2.name + " (" + str(obj2.uuid) + ")")

                # Add self
                out.append(obj)

                # Add children
                out.extend(obj.getAllContainedObjectsRecursive(respectContainerStatus))

        # Return
        return out

    # Get one or more objects of a specific type.
    # maxMatches specifies the maximum number of objects to return (all returns are a list)
    # recursive specifies whether to look recursively through all contained objects (True), or just the immediate contents (False)
    # respectContainerStatus specifies whether to respect the container status (e.g. if it's closed, then don't return anything, or recurse into it)
    def getContainedObjectByType(self, objType, maxMatches=1, recursive=False, respectContainerStatus=False):
        # Get an object of a specific type
        out = []

        # If not recursive, then just look at the immediate contents.
        if (not recursive):
            if (not respectContainerStatus) or (respectContainerStatus and self.attributes['isOpenContainer']):
                for obj in self.contents:
                    # If this is the type we're looking for, then add it
                    if (obj.type == objType):
                        out.append(obj)
                        if (len(out) >= maxMatches):
                            break

        else:
            # Recursive
            # If this is a container, and it's open, then add the contents
            allObjs = self.getAllContainedObjectsRecursive(respectContainerStatus)
            for obj in allObjs:
                # If this is the type we're looking for, then add it
                if (obj.type == objType):
                    out.append(obj)
                    if (len(out) >= maxMatches):
                        break
        # Return
        return out

    # Get all contained objects and parts, do not respect containers (this is typically used for dumping the objects to a file)
    def getAllContainedObjectsAndParts(self, includeContents=True, includeParts=True):
        # TODO: Fix this so it doesn't require the list(set()) hack (which prevents the duplicates, but is slow)
        out = []
        # Add self
        out.append(self)

        if (includeContents):
            for obj in self.contents:
                # Add self
                out.append(obj)

                # Add children
                out.extend(obj.getAllContainedObjectsAndParts(includeContents, includeParts))

        if (includeParts):
            for obj in self.parts:
                # Add self
                out.append(obj)
                # Add children
                out.extend(obj.getAllContainedObjectsAndParts(includeContents, includeParts))

        # Return
        return list(set(out))

    # Get outermost closed container (if this object is contained in a closed container).
    # Essentially answers the question: "What is the next container that I'd have to open to (eventually) get at this object?"
    # Returns None if there is no closed container.
    def getOutermostClosedContainer(self, includeSelf=False):
        lastContainer = None
        lastParent = self.parentContainer

        # If 'includeSelf' is true, and this object is a closed container, then add it as the initial last container
        if (includeSelf):
            if (self.attributes['isContainer'] == True) and (self.attributes['isOpenContainer'] == False):
                lastContainer = self

        # Iterate through all parents, looking for the last closed container
        while (lastParent != None):
            if (lastParent.attributes['isOpenContainer'] == False):
                lastContainer = lastParent
            lastParent = lastParent.parentContainer

        # Return the last container
        return lastContainer


    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        # Use this object on the patient object
        return ActionSuccess(False, "I'm not sure how to use the " + self.name + " with the " + patientObj.name + ".")

    # Specific use-with actions
    def useWithShovelResult():
        # Use this object with a shovel
        return None


    #
    #   Update/Tick
    #
    def tick(self):
        # Update the object

        # Stop if the object has already had tick() called this update -- this might have happened if the object moved locations in this current update cycle.
        if (self.tickCompleted):
            return

        # Remove all current sprite modifiers (these are regenerated every tick)
        self.curSpriteModifiers = set()

        # Infer current name
        if (self.firstInit):
            self.firstInit = False
            self.inferSpriteName(force=True)
        else:
            self.inferSpriteName()

        # Remove the list of object description modifiers, which should be regenerated every tick.
        self.attributes['microscopeModifierText'] = []

        # Set that the tick has been completed on this object
        self.tickCompleted = True

        # Also call tick on all contained objects
        for obj in self.contents:
            # First, update the grid location of the contained objects to be the same as the container
            obj.setWorldLocation(self.attributes["gridX"], self.attributes["gridY"])
            # Next, call tick()
            obj.tick()
        # And all parts
        for obj in self.parts:
            # First, update the grid location of the contained objects to be the same as the container
            obj.setWorldLocation(self.attributes["gridX"], self.attributes["gridY"])
            # Next, call tick()
            obj.tick()

    # Reset the tick flag for this object (and all contained objects and parts)
    def resetTick(self):
        # Reset the tick flag for this object
        self.tickCompleted = False

        # Also reset the tick flag for all contained objects
        for obj in self.contents:
            obj.resetTick()
        # And all parts
        for obj in self.parts:
            obj.resetTick()


    #
    #   Text Observations
    #
    def getTextDescription(self):
        # Get a text description of this object
        outStr = self.name + self._getContainerTextDescription()
        return outStr

    # Helper intended to write "... (in <containerName> (uuid))" at the end of text descriptions
    def _getContainerTextDescription(self):
        outStr = ""
        if (self.parentContainer != None):
            if (self.world.liveUserPlaying):
                containerName = self.parentContainer.name
            else:
                containerName = self.parentContainer.name + ( " [uuid: " + str(self.parentContainer.uuid) + "]")
            containerPrefix = self.parentContainer.attributes.get("containerPrefix", "in")
            outStr += " (" + containerPrefix + " " + containerName + ")"

        return outStr



    def getTextObservationMicroscopic(self):
        # Get a text description of this object, under a microscope
        # TODO: Check object size (normal, microscopic, nanoscopic?)

        microscopeDescriptionStr = ""

        # First, describe the object itself
        partNames = [part.name for part in self.parts]
        if (len(partNames) > 0):
            microscopeDescriptionStr = "Observing the " + self.name + ", you see the following parts: " + ", ".join(partNames) + ". \n"

        # Describe the materials
        materials = self.attributes['materials']
        if (len(materials) == 0):
            microscopeDescriptionStr += "Observing the " + self.name + ", you don't observe anything particularly noteworthy."
        else:
            microscopeDescriptionStr += "Observing the " + self.name + ", you see the following: "
            materialDescs = []
            for material in materials:
                materialDescs.append(material['microscopeDesc'])
            microscopeDescriptionStr += ", ".join(materialDescs) + ".\n"

        # Add any specific attributes to be mentioned.
        if (len(self.attributes['microscopeModifierText']) > 0):
            microscopeDescriptionStr += " ".join(self.attributes['microscopeModifierText']) + "\n"

        # Then, recursively describe all the parts.
        for part in self.parts:
            microscopeDescriptionStr += part.getTextObservationMicroscopic()

        # Return
        return microscopeDescriptionStr



    #
    #   Sprite
    #

    # Invalidate the sprite name for this object, and all objects it contains/that contain it.
    # This will force them to update their sprites on the next tick.
    def invalidateSpritesThisWorldTile(self):
        #print("##### INVALIDATING SPRITES FOR " + self.name + " #####")
        #print("#### LOCATION: " + str(self.attributes["gridX"]) + ", " + str(self.attributes["gridY"]) + " ####")
        # Step 1: Get the names of all objects (and any objects they contain) at this world tile
        allObjs = self.world.getObjectsAt(self.attributes["gridX"], self.attributes["gridY"], respectContainerStatus=False)
        #print("#### OBJECTS: " + str(allObjs) + " ####")
        # Step 2: Invalidate the sprite names for all objects at this world tile
        for obj in allObjs:
            obj.needsSpriteNameUpdate = True
            #for containedObj in obj.getAllContainedObjectsRecursive():
            #    containedObj.needsSpriteNameUpdate = True

    def inferSpriteName(self, force:bool=False):
        # Infer the sprite name from the current state of the object
        # This should be called whenever the object is moved, or something else changes
        if (self.needsSpriteNameUpdate):
            # Placeholder for inferring sprite name based on attributes
            self.curSpriteName = self.defaultSpriteName
        else:
            self.curSpriteName = self.defaultSpriteName

    def getSpriteName(self):
        # Get the current sprite name, in response to the current state of the object

        # Check if this sprite is invalidated and needs to be updated
        if self.needsSpriteNameUpdate:
            # If so, then update it
            self.inferSpriteName()
            self.needsSpriteNameUpdate = False

        return self.curSpriteName

    # Should be run once per frame, after the rendering for every tile is finished, after the backbuffer flips.
    def updateLastSpriteName(self):
        # Update the last sprite name
        self.lastSpriteName = self.tempLastSpriteName

    def listWithOffset(self, contents):
        positions = []
        if len(contents) == 1:
            positions.append((0, 0))
        elif len(contents) == 2:
            positions.append((-4, 8))
            positions.append((6, -2))
        elif len(contents) == 3:
            positions.append((-4, -4))
            positions.append((6, 2))
            positions.append((-4, 8))
        elif len(contents) == 4:
            positions.append((-4, -4))
            positions.append((4, 0))
            positions.append((-4, 6))
            positions.append((4, 8))
        else:
            def _circle():
                d = 3
                for i in range(len(contents)):
                    t = (i+0.3) / d * np.pi
                    x = 8 * math.cos(t)
                    y = 8 * math.sin(t)
                    yield x, y

            positions = list(_circle())

        # Sort position by y.
        positions.sort(key=lambda x: x[1])

        yield from zip(positions, contents)

    # Now returns a list of dicts ({"spriteName": ..., "yOffset": ...})
    def getContentsSpriteNames(self, yOffset:int=0, xOffset:int=0):
        spriteList = []
        # Then, add the name of any visible contents
        # First, check if this is a container (and it's open)
        if (self.attributes['isContainer'] and self.attributes['isOpenContainer']):
            # Make sure that the sprites for the contents should be displayed, and that this isn't handled by the sprite function rendering different sprites for full vs empty objects
            if (self.attributes['contentsVisible2D']):
                # If so, then add the contents
                for (offX, offY), obj in self.listWithOffset(self.contents):
                    # Add the sprite name of the object
                    spriteNameObj = obj.getSpriteName()

                    if spriteNameObj is not None:
                        # Sprite names of contents
                        spriteList.extend(
                            obj.getSpriteNamesWithContents(
                                xOffset=xOffset + obj.attributes["screenXOffset"] + offX,
                                yOffset=yOffset + obj.attributes["screenYOffset"] + offY,
                            )
                        )

            # If the object has the 'objectToShow' attribute, like for agents showing the last object they interacted with, then add that object's sprite name
            if ('objectToShow' in self.attributes) and (self.attributes['objectToShow'] != None):
                # Get the object to show
                obj = self.attributes['objectToShow']

                # Make sure that object's parent container is this object, otherwise discontinue
                if (obj.parentContainer == self):
                    # Add the sprite name of the object
                    spriteNameObj = obj.getSpriteName()
                    if spriteNameObj is not None:
                        #spriteList.append(spriteNameObj)
                        spriteList.append({"spriteName": "instruments2_box", "yOffset": yOffset - 4, "xOffset": xOffset + 12, "scale": 0.75})
                        spriteList.append({"spriteName": spriteNameObj, "yOffset": yOffset - 4, "xOffset": xOffset + 12, "scale": 0.75})
                        # Add any sprite modifiers
                        #spriteList.extend(obj.curSpriteModifiers)
                        for spriteModifier in obj.curSpriteModifiers:
                            spriteList.append({"spriteName": spriteModifier, "yOffset": yOffset - 4, "xOffset": xOffset + 12, "scale": 0.75})


        return spriteList

    # Now returns a list of dicts ({"spriteName": ..., "yOffset": ...})
    def getSpriteNames(self, yOffset:int=0, xOffset:int=0):
        # First, get the name of the current object itself
        spriteNameOrNames = self.getSpriteName()
        spriteList = []
        if (isinstance(spriteNameOrNames, list)):
            #spriteList.extend(spriteNameOrNames)
            for spriteName in spriteNameOrNames:
                spriteList.append({"spriteName": spriteName, "yOffset": yOffset, "xOffset": xOffset})
        else:
            #spriteList.append(spriteNameOrNames)
            spriteList.append({"spriteName": spriteNameOrNames, "yOffset": yOffset, "xOffset": xOffset})

        # Add any sprite modifiers
        #spriteList.extend(self.curSpriteModifiers)
        for spriteModifier in self.curSpriteModifiers:
            spriteList.append({"spriteName": spriteModifier, "yOffset": yOffset, "xOffset": xOffset})

        return spriteList

    def getSpriteNamesWithContents(self, yOffset:int=0, xOffset:int=0):
        # Get the sprite name, including the contents of the object
        # This is used for rendering objects that contain other objects (e.g. containers)

        # First, get the name of the current object itself
        spriteList = self.getSpriteNames(yOffset=yOffset, xOffset=xOffset)

        # Then, add the name of any visible contents
        spriteList.extend(
            self.getContentsSpriteNames(
                yOffset=yOffset + self.attributes["screenYOffset"],
                xOffset=xOffset + self.attributes["screenXOffset"]
            )
        )

        # Return the sprite list
        return spriteList

    #
    #   Serialize to JSON (for saving histories, but potentially lossy -- not all object member variables are saved)
    #
    def to_dict(self):
        # Serialize to a dictionary suitable for saving to JSON
        # Note: This is lossy, and does not save all member variables
        packed = {
            "uuid": self.uuid,
            "name": self.name,
            "type": self.type,
            "contents": [],
            "parts": [],
            "attributes": {},
            "actionHistory": None,
            "spriteNames": self.getSpriteNamesWithContents()       ## TODO: Deprecated for the moment, since objects now render themselves.
        }

        # Serialize contents
        for obj in self.contents:
            packed["contents"].append( {"objUUID": obj.uuid} )

        # Serialize parts
        for obj in self.parts:
            packed["parts"].append( {"objUUID": obj.uuid} )

        # Serialize attributes
        for key in self.attributes:
            value = self.attributes[key]
            # Skip any non-primitive types (but allow lists and dicts)
            if type(value) not in (str, int, float, bool, list, dict, set):
                continue
            packed["attributes"][key] = value


        # Serialize action history, if applicable
        if (self.actionHistory != None):
            packed["actionHistory"] = self.actionHistory.exportToJSONAbleList()

        return packed
