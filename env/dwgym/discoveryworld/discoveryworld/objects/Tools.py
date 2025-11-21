import numpy as np
from termcolor import colored
from discoveryworld.ActionSuccess import ActionSuccess, MessageImportance
from discoveryworld.objects.Object import Object
from discoveryworld.objects.Terrain import Sand


class Flag(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "flag", "flag", defaultSpriteName = "instruments_flag")

        # Material
        self.attributes["manualMaterialNames"] = ["Paper"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class FlowerPot(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "flowerpot", "flowerpot", defaultSpriteName = "generated_flowerpot")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["PlantMatterGeneric"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class Jar(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "jar", "jar", defaultSpriteName = "placeholder_jar_empty")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")
        self.attributes['contentsVisible2D'] = False               # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)

        self.attributes["manualMaterialNames"] = ["Glass"]

        # Auto fill
        self.autoFillCheckForObjectName = None
        self.autoFillFillObjectName = None
        self.lastAddedCount = 0
        self.replenishTime = 5

    # If we want this object to autofill (i.e. perpetually fill with a supply of an object), set it here.
    def setAutoFill(self, checkObjectName:str, fillObjectName:str, minCount:int, replenishTime:int=8):
        self.autoFillCheckForObjectName = checkObjectName
        self.autoFillFillObjectName = fillObjectName
        self.autoFillMinCount = minCount
        self.replenishTime = replenishTime
        self.lastAddedCount = 0


    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []

        # Add whether it's open
        postModifier = ""
        if (len(self.contents) == 0):
            addedProperties.append("empty")
        else:
            postModifier = " containing items"

        outStr = " ".join(addedProperties) + " " + self.name + postModifier + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr


    def tick(self):
        # Call superclass
        Object.tick(self)

        # Auto fill
        #if (self.autoFillObjectName != None and self.autoFillMinCount != None):
        if (self.autoFillCheckForObjectName != None) and (self.autoFillFillObjectName != None) and (self.autoFillMinCount != None):
            # Count how much of the object is contained within the root level of the contents
            count = 0
            for obj in self.contents:
                if (obj.name == self.autoFillCheckForObjectName):
                    count += 1

            # Fill up to the minimum count
            if (count < self.autoFillMinCount):
                # Check to see if the last added count is zero
                if (self.lastAddedCount <= 0):
                    # Add one more
                    objToAdd = self.world.createObject(self.autoFillFillObjectName)
                    self.addObject(objToAdd, force=True)
                    # Tick the object to update its sprite
                    objToAdd.tick()
                    # Invalidate object sprite
                    self.needsSpriteNameUpdate = True
                    # Tick
                    self.lastAddedCount = self.replenishTime     # Wait a few ticks before adding the next one
                else:
                    self.lastAddedCount -= 1

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return
        # Infer sprite based on whether empty/non-empty
        if (len(self.contents) == 0):
            self.curSpriteName = "placeholder_jar_empty"
        elif (len(self.contents) == 1):
            self.curSpriteName = "placeholder_jar_full1"
        elif (len(self.contents) == 2):
            self.curSpriteName = "placeholder_jar_full2"
        else:
            self.curSpriteName = "placeholder_jar_full3"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName



class PaintBucket(Object):
    # Constructor
    def __init__(self, world, color):
        super().__init__(world, "paint bucket", f"{color} paint bucket", defaultSpriteName = "placeholder_pot_empty")
        self.color = color

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this
        self.attributes['isContainer'] = False                    # Is it a container?

    def tick(self):
        super().tick()
        self.curSpriteModifiers.add(f"placeholder_paint_{self.color}")


class Pot(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "pot", "pot", defaultSpriteName = "placeholder_pot_empty")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")
        self.attributes['contentsVisible2D'] = False               # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    # attributes['temperatureC']
    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []
        if (self.attributes['temperatureC'] <= 10):
            addedProperties.append("cold")
        elif (self.attributes['temperatureC'] > 10) and (self.attributes['temperatureC'] < 100):
            addedProperties.append("warm")
        elif (self.attributes['temperatureC'] >= 100):
            addedProperties.append("hot")

        outStr = " ".join(addedProperties) + " " + self.name + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr


    def tick(self):
        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return
        # Infer sprite based on whether empty/non-empty
        if (len(self.contents) == 0):
            self.curSpriteName = "placeholder_pot_empty"
        elif (len(self.contents) == 1):
            self.curSpriteName = "placeholder_pot_full1"
        elif (len(self.contents) == 2):
            self.curSpriteName = "placeholder_pot_full2"
        else:
            self.curSpriteName = "placeholder_pot_full3"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Shovel(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "shovel", "shovel", defaultSpriteName = "placeholder_shovel")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        self.attributes["isUsable"] = True                        # Can it be used?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        # Use this object on the patient object

        # Check if the patient object has the 'isShovelable' attribute
        if (not patientObj.attributes["isShovelable"]):
            # Can't use the shovel on this object
            return ActionSuccess(False, "You can't use the shovel on that.")


        # Run the 'useWithShovelResult' function on the patient object
        result = patientObj.useWithShovelResult()

        # Return the result
        return result


    def tick(self):
        # Call superclass
        Object.tick(self)


class Coin(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "coin", "coin", defaultSpriteName = "house1_coin")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

class FlagPole(Object):
    # Constructor
    def __init__(self, world, height=1, current_height=0):
        # Default sprite name
        Object.__init__(self, world, "flag", "flag", defaultSpriteName = "instruments2_flag_pole_bottom")

        self.height = height
        self.current_height = current_height

        self.attributes["isMovable"] = False
        self.attributes["isPassable"] = False

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]


    def getSpriteNamesWithContents(self, yOffset:int=0, xOffset:int=0):
        spriteList = super().getSpriteNamesWithContents(yOffset, xOffset)

        nb_flags = int(np.ceil(self.height / 2)) -1

        for i in range(1, nb_flags+1):
            if i == nb_flags:
                spriteList.append({"spriteName": "instruments2_flag_pole_top", "yOffset": yOffset-32*i, "xOffset": xOffset})
            else:
                spriteList.append({"spriteName": "instruments2_flag_pole_middle", "yOffset": yOffset-32*i, "xOffset": xOffset})

        if self.current_height % 2 == 0:
            spriteList.append({"spriteName": "instruments2_flag_bottom", "yOffset": yOffset-32*(self.current_height//2), "xOffset": xOffset})
        else:
            spriteList.append({"spriteName": "instruments2_flag_top", "yOffset": yOffset-32*((self.current_height-1)//2), "xOffset": xOffset})

        return spriteList


class MeasuringTape(Object):
    def __init__(self, world):
        super().__init__(world, "tape", "measuring tape", defaultSpriteName="instruments2_measuring_tape")
        self.attributes['isUsable'] = True  # Can this device be used with another object? (e.g. specifically through the 'use' action)

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def actionUseWith(self, otherObject=None):
        if isinstance(otherObject, FlagPole):
            flagpole = otherObject
            useDescriptionStr = f"The flag is {flagpole.current_height} meter from the ground.\n"

            if flagpole.current_height >= 2:
                useDescriptionStr = useDescriptionStr.replace("meter", "meters")

        else:
            useDescriptionStr = "You need a measuring device first.\n"
            return ActionSuccess(False, useDescriptionStr, importance=MessageImportance.LOW)

        return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.LOW)

class ColoredKey(Object):

    def __init__(self, world, color):
        super().__init__(world, "key", f"{color} key", defaultSpriteName=f"instruments_key_{color}")
        self.color = color

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this
        self.attributes['isContainer'] = False                    # Is it a container?

    def tick(self):
        super().tick()
        self.curSpriteModifiers.add(f"instruments_key_{self.color}")

class SpeedSquare(Object):
    def __init__(self, world):
        super().__init__(world, "speed square", "speed square", defaultSpriteName="instruments2_speed_square")
        self.attributes['isUsable'] = True
        self.collectedMeasurements = set()

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def actionUseWith(self, otherObject=None):
        objects = self.world.getObjectsAt(*self.getWorldLocation())
        for obj in objects:
            if "lightAngle" in obj.attributes:
                self.collectedMeasurements.add(obj.attributes["lightAngle"])
                useDescriptionStr = f"The rays coming from Planet X's star are {obj.attributes['lightAngle']} millidegree from the ground at the current location.\n"

                if obj.attributes['lightAngle'] >= 2:
                    useDescriptionStr = useDescriptionStr.replace("degree", "degrees")

                return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

        useDescriptionStr = "Use on open sky ground tile to measure light angle.\n"
        return ActionSuccess(False, useDescriptionStr, importance=MessageImportance.LOW)



class Pendulum(Object):
    def __init__(self, world, part=None):
        # assert part in ["top", "bottom"]
        # super().__init__(world, "pendulum", f"pendulum ({part})", defaultSpriteName=f"instruments2_pendulum_{part}")
        super().__init__(world, "pendulum", f"pendulum", defaultSpriteName=f"instruments2_pendulum")
        self.attributes['isActivatable'] = True
        self.attributes['isActivated'] = False
        self.attributes['isPassable'] = False
        self.attributes['isReadable'] = True
        self.nbTicksSinceActivation = 0
        self.oscillationPeriod = 7
        self.length = 1

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def setLength(self, length):
        self.length = length
        self.name = f"{length}-meter pendulum"

    def tick(self):
        super().tick()
        if self.attributes['isActivated']:
            self.nbTicksSinceActivation += 1
        else:
            self.nbTicksSinceActivation = 0

        # Compute the number of oscillations since activation.
        # nbOscillations = self.nbTicksSinceActivation // self.oscillationPeriod
        nbOscillations = np.round(self.nbTicksSinceActivation / self.oscillationPeriod, 3)
        self.attributes["document"] = f"{nbOscillations} oscillations in {self.nbTicksSinceActivation} ticks."

    def getTextDescription(self):
        status = "activated" if self.attributes['isActivated'] else "deactivated"
        return f"{self.name} ({status}) => {self.attributes['document']}"
