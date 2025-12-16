from discoveryworld.Layer import Layer


def mkCaveOneRoom(world, x, y, width, height, signText="Default Sign Text"):
    includeDoor = True

    # Walls
    # Sprite names: ['house1_house_corner_b', 'house1_house_corner_bl', 'house1_house_corner_br', 'house1_house_corner_l', 'house1_house_corner_r', 'house1_house_corner_t', 'house1_house_corner_tl', 'house1_house_corner_tr']
    # Check that it has a minimum size (e.g. at least 4x4)
    if width < 4 or height < 4:
        print("Error: Cave is too small: " + str(width) + ", " + str(height))
        return

    # Floor
    for i in range(0, width):
        for j in range(0, height):
            world.addObject(x + i, y + j, Layer.WORLD, world.createObject("CaveFloor"))

    # Top-left corner
    world.addObject(x, y, Layer.BUILDING, world.createObject("CaveWall"))
    # Top-right corner
    world.addObject(x + width - 1, y, Layer.BUILDING, world.createObject("CaveWall"))
    # Bottom-left corner
    world.addObject(x, y + height - 1, Layer.BUILDING, world.createObject("CaveWall"))
    # Bottom-right corner
    world.addObject(x + width - 1, y + height - 1, Layer.BUILDING, world.createObject("CaveWall"))
    # Top wall
    for i in range(1, width - 1):
        world.addObject(x + i, y, Layer.BUILDING, world.createObject("CaveWall"))
    # Bottom wall
    for i in range(1, width - 1):
        # The middle of the bottom wall should be a door
        if (i == int(width / 2) and includeDoor):
            world.addObject(x + i, y + height - 1, Layer.BUILDING, world.createObject("CaveFloor"))
            #world.addObject(x + i, y + height - 1, Layer.FURNITURE, world.createObject("Door"))
        else:
            world.addObject(x + i, y + height - 1, Layer.BUILDING, world.createObject("CaveWall"))
    # Left wall
    for i in range(1, height - 1):
        world.addObject(x, y + i, Layer.BUILDING, world.createObject("CaveWall"))
    # Right wall
    for i in range(1, height - 1):
        world.addObject(x + width - 1, y + i, Layer.BUILDING, world.createObject("CaveWall"))
    # Sign infront of the door
    if (len(signText) > 0):
        sign = world.createObject("Sign")
        sign.setText(signText)
        world.addObject(x + int(width / 2)-1, y + height, Layer.FURNITURE, sign)


def mkCave(x, y, world):
    # Make the cave
    mkCaveOneRoom(world, x, y, 6, 5, signText="Cave")

    # Add some rocks
    world.addObject(x+1, y+1, Layer.OBJECTS, world.createObject("rock_lg"))
    world.addObject(x+4, y+1, Layer.OBJECTS, world.createObject("rock_med"))
    world.addObject(x+4, y+2, Layer.OBJECTS, world.createObject("rock_lg"))
    world.addObject(x+4, y+3, Layer.OBJECTS, world.createObject("rock_sm"))
    world.addObject(x+3, y+1, Layer.OBJECTS, world.createObject("rock_sm"))

    world.addObject(x+1, y+2, Layer.OBJECTS, world.createObject("rock_glowing"))
    world.addObject(x+2, y+1, Layer.OBJECTS, world.createObject("rock_glowing"))


def mkCaveChallenge(x, y, world):
    glowingRocks = []
    # Make the cave
    mkCaveOneRoom(world, x, y, 6, 5, signText="Cave")

    # Add some rocks
    world.addObject(x+1, y+1, Layer.OBJECTS, world.createObject("rock_lg"))
    world.addObject(x+4, y+1, Layer.OBJECTS, world.createObject("rock_med"))
    world.addObject(x+4, y+2, Layer.OBJECTS, world.createObject("rock_lg"))
    world.addObject(x+4, y+3, Layer.OBJECTS, world.createObject("rock_sm"))
    world.addObject(x+3, y+1, Layer.OBJECTS, world.createObject("rock_sm"))

    glowingRock1 = world.createObject("GlowingRockDetector")
    world.addObject(x+1, y+2, Layer.OBJECTS, glowingRock1)
    glowingRocks.append(glowingRock1)
    glowingRock2 = world.createObject("GlowingRockDetector")
    world.addObject(x+2, y+1, Layer.OBJECTS, glowingRock2)
    glowingRocks.append(glowingRock2)

    return glowingRocks
