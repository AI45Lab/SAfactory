

# Archaeology Dig Site
from discoveryworld.Layer import Layer


def mkDigSite(x, y, world, r, digSiteNum, artifactAge):
    minSize = 2
    # Randomly make size of archaeology dig site (x-y each between 1-2)
    totalLocations = 0
    while (totalLocations < minSize):       # Keep trying until we get a big enough dig site (at least 2 squares)
        digSiteSizeX = r.randint(1, 2)
        digSiteSizeY = r.randint(1, 2)
        totalLocations = digSiteSizeX * digSiteSizeY
    locationOfArtifact = r.randint(0, totalLocations-1)

    # The artifact to add
    artifact = world.createObject("AncientArtifact")
    # Add the age
    artifact.attributes['radiocarbonAge'] = artifactAge

    # Make the dig site soil (with one hole containing an artifact)
    count = 0       # Counter for where to place the artifact
    for i in range(0, digSiteSizeX):
        for j in range(0, digSiteSizeY):
            soilTile = world.createObject("SoilTile")
            # Randomly set the 'hasHole' attribute to True for some of the soil tiles
            #if (random.randint(0, 2) == 0):
            #    soilTile.attributes['hasHole'] = True

            # Randomly add an artifact to one of the soil tiles
            if (count == locationOfArtifact):
                # Add the artifact to the hole
                soilTile.addObject(artifact, force=True)

            # Add soil tile to world
            world.addObject(x+i, y+j+1, Layer.BUILDING, soilTile)

            count += 1

    # Add a sign to the dig site
    sign = world.createObject("Sign")
    sign.setText("Dig Site " + str(digSiteNum))
    world.addObject(x, y, Layer.FURNITURE, sign)

    # Return the artifact that was added
    return artifact, sign



# Archaeology Dig Site
def mkDigSiteWithObj(x, y, world, r, digSiteNum, artifactObj, artifactExposed=False):
    minSize = 2
    # Randomly make size of archaeology dig site (x-y each between 1-2)
    totalLocations = 0
    while (totalLocations < minSize):       # Keep trying until we get a big enough dig site (at least 2 squares)
        digSiteSizeX = r.randint(1, 2)
        digSiteSizeY = r.randint(1, 2)
        totalLocations = digSiteSizeX * digSiteSizeY
    locationOfArtifact = r.randint(0, totalLocations-1)

    # Make the dig site soil (with one hole containing an artifact)
    count = 0       # Counter for where to place the artifact
    for i in range(0, digSiteSizeX):
        for j in range(0, digSiteSizeY):
            soilTile = world.createObject("SoilTile")
            # Randomly set the 'hasHole' attribute to True for some of the soil tiles
            #if (random.randint(0, 2) == 0):
            #    soilTile.attributes['hasHole'] = True

            # Randomly add an artifact to one of the soil tiles
            if (count == locationOfArtifact):
                # If 'artifactExposed' is True, then remove the soil from the hole
                if (artifactExposed):
                    for obj in soilTile.contents:
                        soilTile.removeObject(obj)

                # Add the artifact to the hole
                soilTile.addObject(artifactObj, force=True)

            # Add soil tile to world
            world.addObject(x+i, y+j+1, Layer.BUILDING, soilTile)

            count += 1


    # Add a sign to the dig site
    sign = world.createObject("Sign")
    sign.setText("Dig Site " + str(digSiteNum))
    world.addObject(x, y, Layer.FURNITURE, sign)

    # Return the sign that was added
    return sign