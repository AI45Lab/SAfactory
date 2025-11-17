from discoveryworld.objects.Object import Object

class CaveFloor(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "floor", "cave floor", defaultSpriteName = "cave1_rock_floor")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Static -- just use the default sprite name
        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class CaveWall(Object):
    # TODO: Most of the interior wall parts of this code are disabled -- so they can either be removed, or the sprite sheet can be modified to include interior walls.

    # Constructor
    def __init__(self, world):
        # Note: Change the default sprite name to something obviously incorrect so it is obvious when it's not inferring properly.
        Object.__init__(self, world, "wall", "cave wall", defaultSpriteName = "cave1_wall_t")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

        # Rendering attribute (wall direction, "tl", "tr", "bl", "br", "l", "r", "t", "b")
        self.wallShape = ""


    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # Call superclass
        Object.tick(self)

    # Helper function to get the neighbouring walls
    def _hasWall(self, x, y):
        # Check to see if there is a wall at the given location
        # First, get objects at a given location
        objects = self.world.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if (object.type == "wall"):
                return (True, object.wallShape)
            elif (object.type == "door"):
                return (True, "")
        # If we get here, there are no walls
        return (False, "")

    # Helper function to get the neighbouring walls
    def _hasFloor(self, x, y):
        # Check to see if there is a wall at the given location
        # First, get objects at a given location
        objects = self.world.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if object.type == "floor":
                return True
        # If we get here, there are no walls
        return False


    # def getSpriteName(self):
    #     # Get the current sprite name, in response to the current state of the object
    #     # Normally this just returns the current sprite name, but for these cave walls (that are partially transparent), we're going to add a background object -- the cave floor.
    #     caveFloor = "cave1_floor"
    #     return [caveFloor, self.curSpriteName]


    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see what the neighbouring walls are
        hasWallNorth, wallShapeNorth = self._hasWall(self.attributes["gridX"], self.attributes["gridY"] - 1)
        hasWallSouth, wallShapeSouth = self._hasWall(self.attributes["gridX"], self.attributes["gridY"] + 1)
        hasWallWest, wallShapeWest = self._hasWall(self.attributes["gridX"] - 1, self.attributes["gridY"])
        hasWallEast, wallShapeEast = self._hasWall(self.attributes["gridX"] + 1, self.attributes["gridY"])

        # Check to see if there is neighbouring floor
        hasFloorNorth = self._hasFloor(self.attributes["gridX"], self.attributes["gridY"] - 1)
        hasFloorSouth = self._hasFloor(self.attributes["gridX"], self.attributes["gridY"] + 1)
        hasFloorWest = self._hasFloor(self.attributes["gridX"] - 1, self.attributes["gridY"])
        hasFloorEast = self._hasFloor(self.attributes["gridX"] + 1, self.attributes["gridY"])

        #isInterior = (hasFloorNorth and hasFloorSouth) or (hasFloorWest and hasFloorEast)
        isInterior = False

        # Corners
        # First, 4 corners
        if (hasWallNorth and hasWallEast and hasWallSouth and hasWallWest):
            # 4-way
            self.curSpriteName = "cave1_wall_int_4way"
            self.wallShape = "4way"

        # Next, 3 corners
        elif (hasWallNorth and hasWallSouth and hasWallEast):
            # 3-way, up/down/right
            self.curSpriteName = "cave1_wall_int_tbr"
            self.wallShape = "tbr"
        elif (hasWallNorth and hasWallSouth and hasWallWest):
            # 3-way, up/down/left
            self.curSpriteName = "cave1_wall_int_tbl"
            self.wallShape = "tbl"
        elif (hasWallEast and hasWallWest and hasWallNorth):
            # 3-way, left/right/up
            self.curSpriteName = "cave1_wall_int_lrt"
            self.wallShape = "lrt"
        elif (hasWallEast and hasWallWest and hasWallSouth):
            # 3-way, left/right/down
            self.curSpriteName = "cave1_wall_int_lrb"
            self.wallShape = "lrb"

        # Next, Corners (NE, NW, SE, SW)
        elif (hasWallSouth and hasWallEast):
            self.curSpriteName = "cave1_wall_tl"
            self.wallShape = "tl"
        elif (hasWallSouth and hasWallWest):
            self.curSpriteName = "cave1_wall_tr"
            self.wallShape = "tr"
        elif (hasWallNorth and hasWallEast):
            self.curSpriteName = "cave1_wall_bl"
            self.wallShape = "bl"
        elif (hasWallNorth and hasWallWest):
            self.curSpriteName = "cave1_wall_br"
            self.wallShape = "br"

        # Next, edges (interior)
        elif (isInterior):
            # If a wall north or south, then it's a vertical wall
            if (hasWallNorth or hasWallSouth):
                if (hasWallNorth and hasWallSouth):
                    self.curSpriteName = "cave1_wall_int_tb"
                    self.wallShape = "v"
                elif (hasWallNorth):
                    self.curSpriteName = "cave1_wall_int_tb_opening_t"
                    self.wallShape = "v"
                elif (hasWallSouth):
                    self.curSpriteName = "cave1_wall_int_tb_opening_b"
                    self.wallShape = "v"
            # If a wall east or west, then it's a horizontal wall
            elif (hasWallEast or hasWallWest):
                if (hasWallEast and hasWallWest):
                    self.curSpriteName = "cave1_wall_int_lr"
                    self.wallShape = "h"
                elif (hasWallEast):
                    self.curSpriteName = "cave1_wall_int_lr_opening_l"
                    self.wallShape = "h"
                elif (hasWallWest):
                    self.curSpriteName = "cave1_wall_int_lr_opening_r"
                    self.wallShape = "h"

        # Next, edges (exterior)
        # First, check to see if there is a wall to the north or south
        elif (hasWallNorth or hasWallSouth):
            # Check to see if there is a floor to the east or west
            if (hasFloorEast):
                self.curSpriteName = "cave1_wall_l"
                self.wallShape = "l"
            elif (hasFloorWest):
                self.curSpriteName = "cave1_wall_r"
                self.wallShape = "r"

        # Next, check to see if there is a wall to the east or west
        elif (hasWallEast or hasWallWest):
            # Check to see if there is a floor to the north or south
            if (hasFloorNorth):
                self.curSpriteName = "cave1_wall_t"
                self.wallShape = "t"
            elif (hasFloorSouth):
                self.curSpriteName = "cave1_wall_b"
                self.wallShape = "b"

        else:
            # Catch-all -- possibly a wall with no neighbours?
            print("Unknown wall shape!")
            self.curSpriteName = self.defaultSpriteName
            self.curSpriteName = "cave1_wall_t"
            self.wallShape = "single"


        # If we reach here and still don't know the wall shape, then we'll note that it should be checked again the next tick
        if (self.wallShape == ""):
            self.needsSpriteNameUpdate = True

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Door(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "door", "door", defaultSpriteName = "house2_door_closed")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?

        # Open/Close attribute (similar to container, but not a container)
        self.attributes['isPassage'] = True                       # Is this a passage?
        self.attributes['isOpenable'] = True                      # Can be opened
        self.attributes['isOpenPassage'] = False                  # If it's a passage, then is it open?
        self.attributes['requiresKey'] = 0                        # If it requires a key to open, then this is a special ID for the key.  If the value is <=0, then it doesn't require a key.

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]


    def setKeyID(self, keyID:int):
        self.attributes['requiresKey'] = keyID

    def getTextDescription(self):
        # Get a text description of this object
        addedProperties = []

        # Add whether it's open or closed
        if (self.attributes['isOpenPassage'] == True):
            addedProperties.append("open")
        else:
            if (self.attributes['requiresKey'] > 0):
                addedProperties.append("closed locked")
            else:
                addedProperties.append("closed")

        outStr = " ".join(addedProperties) + " " + self.name + self._getContainerTextDescription()
        outStr = outStr.strip()
        return outStr



    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO
        # Randomly open/close the door
        #if (random.randint(0, 100) < 5):
        #    self.attributes["isOpenPassage"] = not self.attributes["isOpenPassage"]
        #    self.needsSpriteNameUpdate = True

        # If the door is open, the object is passable.  If closed, impassable.
        if (self.attributes["isOpenPassage"]):
            self.attributes["isPassable"] = True
        else:
            self.attributes["isPassable"] = False

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If the door is open, then we need to use the open sprite
        if (self.attributes["isOpenPassage"]):
            self.curSpriteName = "house2_door_open"
        else:
            self.curSpriteName = "house2_door_closed"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Fence(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "fence", "fence", defaultSpriteName = "village1_fence_single")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

    def tick(self):
        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see if the neighbouring tiles have paths
        hasFenceNorth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] - 1, "fence")
        hasFenceSouth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] + 1, "fence")
        hasFenceWest = self.world.hasObj(self.attributes["gridX"] - 1, self.attributes["gridY"], "fence")
        hasFenceEast = self.world.hasObj(self.attributes["gridX"] + 1, self.attributes["gridY"], "fence")

        # 4 positives (should never happen unless something is wonky -- just in case, place a single fence)
        if (hasFenceNorth and hasFenceSouth and hasFenceEast and hasFenceWest):
            self.curSpriteName = "village1_fence_single"

        # 3 positives
        # elif (hasFenceNorth and hasFenceSouth and hasFenceEast):
        #     self.curSpriteName = "village1_fence_l"
        # elif (hasFenceNorth and hasFenceSouth and hasFenceWest):
        #     self.curSpriteName = "village1_fence_r"
        # elif (hasFenceEast and hasFenceWest and hasFenceNorth):
        #     self.curSpriteName = "village1_fence_b"
        # elif (hasFenceEast and hasFenceWest and hasFenceSouth):
        #     self.curSpriteName = "village1_fence_t"

        # 2 positives (up/down or left/right)
        elif (hasFenceNorth and hasFenceSouth):
            self.curSpriteName = "village1_fence_l"
        elif (hasFenceEast and hasFenceWest):
            self.curSpriteName = "village1_fence_t"

        # 2 positives (top/left or top/right or bottom/left or bottom/right)
        elif (hasFenceNorth and hasFenceWest):
            self.curSpriteName = "village1_fence_br"
        elif (hasFenceNorth and hasFenceEast):
            self.curSpriteName = "village1_fence_bl"
        elif (hasFenceSouth and hasFenceWest):
            self.curSpriteName = "village1_fence_tr"
        elif (hasFenceSouth and hasFenceEast):
            self.curSpriteName = "village1_fence_tl"

        # 1 positive (north/east/south/west)
        elif (hasFenceNorth):
            self.curSpriteName = "village1_fence_1way_t"
        elif (hasFenceEast):
            self.curSpriteName = "village1_fence_1way_r"
        elif (hasFenceSouth):
            self.curSpriteName = "village1_fence_1way_b"
        elif (hasFenceWest):
            self.curSpriteName = "village1_fence_1way_l"

        else:
            # If we get here, then we have no path nearby (north/east/south/west)
            self.curSpriteName = "village1_fence_single"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Floor(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "floor", "floor", defaultSpriteName = "house2_floor")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        self.attributes["manualMaterialNames"] = ["FloorWood"]

        # Material type?

    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Static -- just use the default sprite name
        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Grass(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "grass", "grass", defaultSpriteName = "forest1_grass")

        self.attributes["isMovable"] = False                       # Can it be moved?

        self.attributes["manualMaterialNames"] = ["PlantMatterGeneric"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class Path(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "path", "path", defaultSpriteName = "forest1_path_c")

        self.attributes["isMovable"] = False                       # Can it be moved?

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

    def tick(self):
        # Call superclass
        Object.tick(self)


    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see if the neighbouring tiles have paths
        hasPathNorth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] - 1, "path")
        hasPathSouth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] + 1, "path")
        hasPathWest = self.world.hasObj(self.attributes["gridX"] - 1, self.attributes["gridY"], "path")
        hasPathEast = self.world.hasObj(self.attributes["gridX"] + 1, self.attributes["gridY"], "path")

        # 4 positives
        if (hasPathNorth and hasPathSouth and hasPathEast and hasPathWest):
            self.curSpriteName = "forest1_path_c"

        # 3 positives
        elif (hasPathNorth and hasPathSouth and hasPathEast):
            self.curSpriteName = "forest1_path_l"
        elif (hasPathNorth and hasPathSouth and hasPathWest):
            self.curSpriteName = "forest1_path_r"
        elif (hasPathEast and hasPathWest and hasPathNorth):
            self.curSpriteName = "forest1_path_b"
        elif (hasPathEast and hasPathWest and hasPathSouth):
            self.curSpriteName = "forest1_path_t"

        # 2 positives (up/down or left/right)
        elif (hasPathNorth and hasPathSouth):
            self.curSpriteName = "forest1_path_lr"
        elif (hasPathEast and hasPathWest):
            self.curSpriteName = "forest1_path_tb"

        # 2 positives (top/left or top/right or bottom/left or bottom/right)
        elif (hasPathNorth and hasPathWest):
            self.curSpriteName = "forest1_path_br"
        elif (hasPathNorth and hasPathEast):
            self.curSpriteName = "forest1_path_bl"
        elif (hasPathSouth and hasPathWest):
            self.curSpriteName = "forest1_path_tr"
        elif (hasPathSouth and hasPathEast):
            self.curSpriteName = "forest1_path_tl"

        # 1 positive (north/east/south/west)
        elif (hasPathNorth):
            self.curSpriteName = "forest1_path_1way_t"
        elif (hasPathEast):
            self.curSpriteName = "forest1_path_1way_r"
        elif (hasPathSouth):
            self.curSpriteName = "forest1_path_1way_b"
        elif (hasPathWest):
            self.curSpriteName = "forest1_path_1way_l"

        else:
            # If we get here, then we have no path nearby (north/east/south/west)
            self.curSpriteName = "forest1_path_single"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Sign(Object):
    # Constructor
    def __init__(self, world, variant=None, text="This is a sign."):
        self.variant = variant
        defaultSpriteName = "village1_sign_writing"
        if self.variant:
            defaultSpriteName = f"village1_sign_writing{self.variant}"
        Object.__init__(self, world, "sign", "sign", defaultSpriteName=defaultSpriteName)

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this object

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

        self.attributes['isReadable'] = True                       # Can it be read?
        self.attributes["document"] = text

    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # Call superclass
        Object.tick(self)

    def setText(self, text:str):
        self.attributes["document"] = text.strip()
        self.needsSpriteNameUpdate = True

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If the sign has text, we choose the sign accordingly.
        if (len(self.attributes["document"]) > 0):
            self.curSpriteName = self.defaultSpriteName
        else:
            self.curSpriteName = "village1_sign_nowriting"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName

        # After one run, we don't need to update the sprite name again unless something changes
        self.needsSpriteNameUpdate = False

    # def render(self, spriteLibrary, window, screenX, screenY, scale):
    #     if self.variant is None:
    #         super().render(spriteLibrary, window, screenX, screenY, scale)
    #     else:
    #         for spriteName in self.getSpriteNamesWithContents():
    #             spriteLibrary.renderSprite(window, spriteName, screenX, screenY-32, scale)


class SignVillage(Object):
    ### TODO: CURRENTLY DOES NOT HANDLE THAT ITS A MULTI-TILE OBJECT

    # Constructor
    def __init__(self, world, part=None):
        name = "sign (village)"
        isPassable = False
        defaultSpriteName = "village1_sign_village"
        if part == "post_right":
            name += f" {part}"
            defaultSpriteName = "village1_sign_village_br"
        elif part == "post_left":
            name += f" {part}"
            defaultSpriteName = "village1_sign_village_bl"
        elif part == "center":
            name += f" {part}"
            defaultSpriteName = "village1_sign_village_mid"
            isPassable = True
        elif part == "banner":
            name += f" {part}"
            defaultSpriteName = "village1_sign_village_top"
            isPassable = True
        else:
            raise NotImplementedError()

        Object.__init__(self, world, "sign (village)", name, defaultSpriteName=defaultSpriteName)

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = isPassable                 # Agen't can't walk over this
        self.attributes["document"] = "This is a sign."

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # Call superclass
        Object.tick(self)

    def setText(self, text:str):
        self.attributes["document"] = text.strip()
        self.needsSpriteNameUpdate = True

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Static sprite -- always the same (currently)
        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName

        # After one run, we don't need to update the sprite name again unless something changes
        self.needsSpriteNameUpdate = False


class Statue(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "statue", "statue", defaultSpriteName = "statue_statue1")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

    # Add readable text (e.g. for a plaque on the statue)
    def addReadableText(self, text):
        self.attributes["isReadable"] = True
        self.attributes['document'] = text

    def tick(self):
        # Call superclass
        Object.tick(self)


class Wall(Object):
    # Constructor
    def __init__(self, world):
        # Note: Change the default sprite name to something obviously incorrect so it is obvious when it's not inferring properly.
        Object.__init__(self, world, "wall", "wall", defaultSpriteName = "house2_wall_t")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        # Material
        self.attributes["manualMaterialNames"] = ["Wood"]

        # Rendering attribute (wall direction, "tl", "tr", "bl", "br", "l", "r", "t", "b")
        self.wallShape = ""


    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True


        # Call superclass
        Object.tick(self)

    # Helper function to get the neighbouring walls
    def _hasWall(self, x, y):
        # Check to see if there is a wall at the given location
        # First, get objects at a given location
        objects = self.world.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if (object.type == "wall"):
                return (True, object.wallShape)
            elif (object.type == "door"):
                return (True, "")
        # If we get here, there are no walls
        return (False, "")

    # Helper function to get the neighbouring walls
    def _hasFloor(self, x, y):
        # Check to see if there is a wall at the given location
        # First, get objects at a given location
        objects = self.world.getObjectsAt(x, y)
        # Then, check to see if any of them are walls
        for object in objects:
            if object.type == "floor":
                return True
        # If we get here, there are no walls
        return False

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see what the neighbouring walls are
        hasWallNorth, wallShapeNorth = self._hasWall(self.attributes["gridX"], self.attributes["gridY"] - 1)
        hasWallSouth, wallShapeSouth = self._hasWall(self.attributes["gridX"], self.attributes["gridY"] + 1)
        hasWallWest, wallShapeWest = self._hasWall(self.attributes["gridX"] - 1, self.attributes["gridY"])
        hasWallEast, wallShapeEast = self._hasWall(self.attributes["gridX"] + 1, self.attributes["gridY"])

        # Check to see if there is neighbouring floor
        hasFloorNorth = self._hasFloor(self.attributes["gridX"], self.attributes["gridY"] - 1)
        hasFloorSouth = self._hasFloor(self.attributes["gridX"], self.attributes["gridY"] + 1)
        hasFloorWest = self._hasFloor(self.attributes["gridX"] - 1, self.attributes["gridY"])
        hasFloorEast = self._hasFloor(self.attributes["gridX"] + 1, self.attributes["gridY"])

        isInterior = (hasFloorNorth and hasFloorSouth) or (hasFloorWest and hasFloorEast)

        # Corners
        # First, 4 corners
        if (hasWallNorth and hasWallEast and hasWallSouth and hasWallWest):
            # 4-way
            self.curSpriteName = "house2_wall_int_4way"
            self.wallShape = "4way"

        # Next, 3 corners
        elif (hasWallNorth and hasWallSouth and hasWallEast):
            # 3-way, up/down/right
            self.curSpriteName = "house2_wall_int_tbr"
            self.wallShape = "tbr"
        elif (hasWallNorth and hasWallSouth and hasWallWest):
            # 3-way, up/down/left
            self.curSpriteName = "house2_wall_int_tbl"
            self.wallShape = "tbl"
        elif (hasWallEast and hasWallWest and hasWallNorth):
            # 3-way, left/right/up
            self.curSpriteName = "house2_wall_int_lrt"
            self.wallShape = "lrt"
        elif (hasWallEast and hasWallWest and hasWallSouth):
            # 3-way, left/right/down
            self.curSpriteName = "house2_wall_int_lrb"
            self.wallShape = "lrb"

        # Next, Corners (NE, NW, SE, SW)
        elif (hasWallSouth and hasWallEast):
            self.curSpriteName = "house2_wall_tl"
            self.wallShape = "tl"
        elif (hasWallSouth and hasWallWest):
            self.curSpriteName = "house2_wall_tr"
            self.wallShape = "tr"
        elif (hasWallNorth and hasWallEast):
            self.curSpriteName = "house2_wall_bl"
            self.wallShape = "bl"
        elif (hasWallNorth and hasWallWest):
            self.curSpriteName = "house2_wall_br"
            self.wallShape = "br"

        # Next, edges (interior)
        elif (isInterior):
            # If a wall north or south, then it's a vertical wall
            if (hasWallNorth or hasWallSouth):
                if (hasWallNorth and hasWallSouth):
                    self.curSpriteName = "house2_wall_int_tb"
                    self.wallShape = "v"
                elif (hasWallNorth):
                    self.curSpriteName = "house2_wall_int_tb_opening_t"
                    self.wallShape = "v"
                elif (hasWallSouth):
                    self.curSpriteName = "house2_wall_int_tb_opening_b"
                    self.wallShape = "v"
            # If a wall east or west, then it's a horizontal wall
            elif (hasWallEast or hasWallWest):
                if (hasWallEast and hasWallWest):
                    self.curSpriteName = "house2_wall_int_lr"
                    self.wallShape = "h"
                elif (hasWallEast):
                    self.curSpriteName = "house2_wall_int_lr_opening_l"
                    self.wallShape = "h"
                elif (hasWallWest):
                    self.curSpriteName = "house2_wall_int_lr_opening_r"
                    self.wallShape = "h"

        # Next, edges (exterior)
        # First, check to see if there is a wall to the north or south
        elif (hasWallNorth or hasWallSouth):
            # Check to see if there is a floor to the east or west
            if (hasFloorEast):
                self.curSpriteName = "house2_wall_l"
                self.wallShape = "l"
            elif (hasFloorWest):
                self.curSpriteName = "house2_wall_r"
                self.wallShape = "r"

        # Next, check to see if there is a wall to the east or west
        elif (hasWallEast or hasWallWest):
            # Check to see if there is a floor to the north or south
            if (hasFloorNorth):
                self.curSpriteName = "house2_wall_t"
                self.wallShape = "t"
            elif (hasFloorSouth):
                self.curSpriteName = "house2_wall_b"
                self.wallShape = "b"

        else:
            # Catch-all -- possibly a wall with no neighbours?
            print("Unknown wall shape!")
            self.curSpriteName = self.defaultSpriteName
            self.curSpriteName = "house2_wall_t"
            self.wallShape = "single"


        # If we reach here and still don't know the wall shape, then we'll note that it should be checked again the next tick
        if (self.wallShape == ""):
            self.needsSpriteNameUpdate = True

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName



class Sand(Object):
    # Constructor
    def __init__(self, world, variant=None):
        # Default sprite name
        defaultSpriteName = "desert1_sand"
        if variant:
            defaultSpriteName = f"desert1_sand_{variant}"

        super().__init__(world, "sand", "sand", defaultSpriteName=defaultSpriteName)

        self.attributes["isMovable"] = False                       # Can it be moved?
        # self.attributes["manualMaterialNames"] = ["PlantMatterGeneric"]

class SandPath(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "path", "path", defaultSpriteName="desert1_path_c")

        self.attributes["isMovable"] = False                       # Can it be moved?

        self.attributes["obscuresObjectsBelow"] = True             # Does it obscure/hide objects on layers below it?

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

    def tick(self):
        # Call superclass
        Object.tick(self)


    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Check to see if the neighbouring tiles have paths
        hasPathNorth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] - 1, "path")
        hasPathSouth = self.world.hasObj(self.attributes["gridX"], self.attributes["gridY"] + 1, "path")
        hasPathWest = self.world.hasObj(self.attributes["gridX"] - 1, self.attributes["gridY"], "path")
        hasPathEast = self.world.hasObj(self.attributes["gridX"] + 1, self.attributes["gridY"], "path")

        # 4 positives
        if (hasPathNorth and hasPathSouth and hasPathEast and hasPathWest):
            self.curSpriteName = "desert1_path_c"

        # 3 positives
        elif (hasPathNorth and hasPathSouth and hasPathEast):
            self.curSpriteName = "desert1_path_l"
        elif (hasPathNorth and hasPathSouth and hasPathWest):
            self.curSpriteName = "desert1_path_r"
        elif (hasPathEast and hasPathWest and hasPathNorth):
            self.curSpriteName = "desert1_path_b"
        elif (hasPathEast and hasPathWest and hasPathSouth):
            self.curSpriteName = "desert1_path_t"

        # 2 positives (up/down or left/right)
        elif (hasPathNorth and hasPathSouth):
            self.curSpriteName = "desert1_path_lr"
        elif (hasPathEast and hasPathWest):
            self.curSpriteName = "desert1_path_tb"

        # 2 positives (top/left or top/right or bottom/left or bottom/right)
        elif (hasPathNorth and hasPathWest):
            self.curSpriteName = "desert1_path_br"
        elif (hasPathNorth and hasPathEast):
            self.curSpriteName = "desert1_path_bl"
        elif (hasPathSouth and hasPathWest):
            self.curSpriteName = "desert1_path_tr"
        elif (hasPathSouth and hasPathEast):
            self.curSpriteName = "desert1_path_tl"

        # 1 positive (north/east/south/west)
        elif (hasPathNorth):
            self.curSpriteName = "desert1_path_1way_t"
        elif (hasPathEast):
            self.curSpriteName = "desert1_path_1way_r"
        elif (hasPathSouth):
            self.curSpriteName = "desert1_path_1way_b"
        elif (hasPathWest):
            self.curSpriteName = "desert1_path_1way_l"

        else:
            # If we get here, then we have no path nearby (north/east/south/west)
            self.curSpriteName = "desert1_path_single"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class LaunchPad(Object):
    # Constructor
    def __init__(self, world, variant=None):
        # Default sprite name
        defaultSpriteName = "launchSite_pad_bl"
        if variant:
            defaultSpriteName = f"launchSite_pad_{variant}"

        super().__init__(world, "launch pad", "launch pad", defaultSpriteName=defaultSpriteName)

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["obscuresObjectsBelow"] = True
        self.attributes["manualMaterialNames"] = ["Concrete"]
