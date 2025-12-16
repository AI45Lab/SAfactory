from discoveryworld.ScienceHelpers import coolObjects, heatObjects
from discoveryworld.objects.Object import Object


class Bed(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "bed", "bed", defaultSpriteName = "house1_bed_ud")
        #Object.__init__(self, world, "bed", "bed", defaultSpriteName = "house1_bed_lr")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]


    def tick(self):
        # Call superclass
        Object.tick(self)


class Bookcase(Object):
    # Constructor
    def __init__(self, world):
        super().__init__(world, "bed", "bed", defaultSpriteName="house1_bookcase")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this
        self.attributes['isContainer'] = False                     # Is it a container?

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

class Chair(Object):
    # Constructor
    def __init__(self, world, curDirection="west"):
        Object.__init__(self, world, "chair", "chair", defaultSpriteName = "house1_chair_l")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container attributes
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Always open
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

        # Rendering attributes
        self.curDirection = curDirection

    # Sprite helper: Look to see if there's a table nearby
    def _hasTable(self, x, y):
        # Check to see if there is a wall at the given location
        # First, get objects at a given location
        objects = self.world.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if object.type == "table":
                return True
        # If we get here, there are no walls
        return False

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see if there is a table north, east, south, or west of us
        hasTableNorth = self._hasTable(self.attributes["gridX"], self.attributes["gridY"] - 1)
        hasTableSouth = self._hasTable(self.attributes["gridX"], self.attributes["gridY"] + 1)
        hasTableEast = self._hasTable(self.attributes["gridX"] + 1, self.attributes["gridY"])
        hasTableWest = self._hasTable(self.attributes["gridX"] - 1, self.attributes["gridY"])

        # If we have a table to the north, then we need to use the north sprite
        if (hasTableNorth):
            self.curSpriteName = "house1_chair_u"
            self.curDirection = "north"
        elif (hasTableSouth):
            self.curSpriteName = "house1_chair_d"
            self.curDirection = "south"
        elif (hasTableEast):
            self.curSpriteName = "house1_chair_r"
            self.curDirection = "east"
        elif (hasTableWest):
            self.curSpriteName = "house1_chair_l"
            self.curDirection = "west"
        else:
            # If we get here, then we have no table nearby
            self.curSpriteName = "house1_chair_l"
            self.curDirection = "west"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Fridge(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "fridge", "fridge", defaultSpriteName = "house1_fridge_closed")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container attributes
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = True                       # Can be opened
        self.attributes['isOpenContainer'] = False                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")

        # Device (is activable)
        self.attributes['isActivatable'] = True                      # Is this a device? (more specifically, can it be activated/deactivated?)
        self.attributes['isActivated'] = True                      # Is this device currently activated?

        # Heating
        self.attributes['coolSourceMinTemp'] = -4                    # If it is a cool source, then this is the minimum temperature that it can reach

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []

        # Add whether it's open
        if (self.attributes['isOpenContainer'] == True):
            addedProperties.append("open")
        else:
            addedProperties.append("closed")

        outStr = " ".join(addedProperties) + " " + self.name + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr

    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # If the fridge is on, then all of its contents gets cooled towards its minimum cooling temperature
        if (self.attributes["isActivated"]):
            # Get a list of all the contents (recursive, with parts)
            contentsAndParts = self.getAllContainedObjectsAndParts(includeContents=True, includeParts=True)
            # Heat them
            coolObjects(contentsAndParts, finalTemperature = self.attributes['coolSourceMinTemp'])


        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = self.defaultSpriteName

        # TODO: If the fridge is open, then we need to use the open sprite
        if (self.attributes["isOpenContainer"]):
            self.curSpriteName = "house1_fridge_open"
        else:
            self.curSpriteName = "house1_fridge_closed"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Sink(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "sink", "sink", defaultSpriteName = "house1_sink_off")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container attributes
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Always open
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")

        # Device (is activable)
        self.attributes['isActivatable'] = True                     # Is this a device? (more specifically, can it be activated/deactivated?)
        self.attributes['isActivated'] = False                      # Is this device currently activated?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []

        # Add whether it's on
        if (self.attributes['isActivated'] == True):
            addedProperties.append("activated")
        else:
            addedProperties.append("inactive")

        outStr = " ".join(addedProperties) + " " + self.name + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr


    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO
        # Randomly turn on/off
        #if (random.randint(0, 100) < 5):
        #    self.attributes["isActivated"] = not self.attributes["isActivated"]
        #    self.needsSpriteNameUpdate = True

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If the stove is open, then we need to use the open sprite
        if (self.attributes["isActivated"]):
            self.curSpriteName = "house1_sink_on"
        else:
            self.curSpriteName = "house1_sink_off"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Stove(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "stove", "stove", defaultSpriteName = "house1_stove_off")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container attributes
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = True                       # Can be opened
        self.attributes['isOpenContainer'] = False                 # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")

        # Device (is activable)
        self.attributes['isActivatable'] = True                       # Is this a device? (more specifically, can it be activated/deactivated?)
        self.attributes['isActivated'] = False                      # Is this device currently activated?

        # Heating
        self.attributes['heatSourceMaxTemp'] = 350                   # If it is a heat source, then this is the maximum temperature that it can reach

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []

        # Add whether it's open
        if (self.attributes['isOpenContainer'] == True):
            addedProperties.append("open")
        else:
            addedProperties.append("closed")

        # Add whether it's on
        if (self.attributes['isActivated'] == True):
            addedProperties.append("activated")
        else:
            addedProperties.append("inactive")

        outStr = " ".join(addedProperties) + " " + self.name + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr


    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # If the stove is on, then all of its contents gets heated towards its maximum heating temperature
        if (self.attributes["isActivated"]):
            # Get a list of all the contents (recursive, with parts)
            contentsAndParts = self.getAllContainedObjectsAndParts(includeContents=True, includeParts=True)
            # Heat them
            heatObjects(contentsAndParts, finalTemperature = self.attributes['heatSourceMaxTemp'])


        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If the stove is open, then we need to use the open sprite
        if (self.attributes["isOpenContainer"]):
            if (self.attributes["isActivated"]):
                self.curSpriteName = "house1_stove_on_open"
            else:
                self.curSpriteName = "house1_stove_open"

        else:
            if (self.attributes["isActivated"]):
                self.curSpriteName = "house1_stove_on"
            else:
                self.curSpriteName = "house1_stove_off"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Table(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "table", "table", defaultSpriteName = "house1_table")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored on the table surface)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

        # Rendering attributes
        self.attributes["screenYOffset"] = -7                      # Small Y offset. This is to make it look like the objects are on the table.


    def tick(self):
        # Call superclass
        Object.tick(self)


class TableBedside(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "table", "table", defaultSpriteName = "house1_table_bedside")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = True                       # Can be opened
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class TableWithSign(Object):
    # Constructor
    def __init__(self, world, signText=""):
        # Default sprite name
        Object.__init__(self, world, "table", "table with a sign", defaultSpriteName = "house1_table")
        self.sign = world.createObject("Sign", variant=2)
        self.sign.setText(signText)
        self.attributes["document"] = signText

        self.attributes["isReadable"] = True                       # Can it be moved?
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored on the table surface)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

        # Rendering attributes
        self.attributes["screenYOffset"] = -7                      # Small Y offset. This is to make it look like the objects are on the table.

    def getSpriteNamesWithContents(self, yOffset:int=0, xOffset:int=0):
        spriteList = super().getSpriteNamesWithContents(yOffset, xOffset)
        spriteList.append({"spriteName": self.sign.getSpriteName(), "yOffset": yOffset+7, "xOffset": xOffset})
        return spriteList


class Desk(Table):
    def __init__(self, world, facing="north"):
        self.facing = facing
        Object.__init__(self, world, "desk", "desk", defaultSpriteName = "house1_table")
        #self.chair = world.createObject("Chair")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored on the table surface)
        self.attributes['isOpenContainer'] = True                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Rendering attributes
        self.attributes["screenYOffset"] = -7                      # Small Y offset. This is to make it look like the objects are on the table.

    def getSpriteNamesWithContents(self, yOffset:int=0, xOffset:int=0):
        spriteList = []

        # Pick the correct chair sprite according to the facing direction of the objects.
        # Depending on the facing direction, we either draw the table on top or not.
        if self.facing == "north":
            spriteList.extend(super().getSpriteNamesWithContents(yOffset, xOffset))
            spriteList.append({"spriteName": "house1_chair_u_back", "xOffset": xOffset, "yOffset": yOffset+7})
        elif self.facing == "south":
            spriteList.append({"spriteName": "house1_chair_d", "xOffset": xOffset, "yOffset": yOffset-10})
            spriteList.extend(super().getSpriteNamesWithContents(yOffset, xOffset))
        elif self.facing == "west":
            spriteList.append({"spriteName": "house1_chair_r", "xOffset": xOffset-16, "yOffset": yOffset-3})
            spriteList.extend(super().getSpriteNamesWithContents(yOffset, xOffset))
        elif self.facing == "east":
            spriteList.append({"spriteName": "house1_chair_l", "xOffset": xOffset+16, "yOffset": yOffset-3})
            spriteList.extend(super().getSpriteNamesWithContents(yOffset, xOffset))

        return spriteList
