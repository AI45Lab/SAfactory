# Sprite Library class
# Keeps a library of sprites loaded in from a number of spritesheets
# Each sprite has a name associated with it (e.g. "table1_left", "grass1_center", "player1_right", etc).
# The name is used to retrieve the sprite from the library.

import json
from os.path import join as pjoin

import pygame

from discoveryworld.constants import ASSETS_PATH


class SpriteLibrary:
    # Constructor
    def __init__(self, assetPath, filenameIndex):
        self.assetPath = assetPath or ASSETS_PATH
        self.filenameIndex = filenameIndex
        self.sprites = {}
        self.warnings = []

        # Load sprites from index
        self.loadIndex(pjoin(self.assetPath, filenameIndex))

        # Print warnings
        if len(self.warnings) > 0:
            print("Warnings (" + str(len(self.warnings)) + "):")
            for idx, warning in enumerate(self.warnings):
                print(str(idx) + ": " + warning)


    # Load sprite sheet index
    # The index is a JSON file that contains a list of spritesheet filenames and the sprites contained in each spritesheet
    def loadIndex(self, filenameIndex):
        print ("Loading sprites from index file: " + filenameIndex)

        # Load the JSON index file
        with open(filenameIndex) as f:
            data = json.load(f)

        # For each spritesheet in the index, load the spritesheet and add the sprites to the library
        count = 0
        for spritesheet in data:
            count += self.loadSpritesheet(spritesheet)

        print("Loaded " + str(count) + " total sprites from " + filenameIndex + ".")


    # Add the sprites from a single spritesheet to the library
    # format:
    # {
    #     "sheetName": "house1",
    #     "filename": "pixymoon/CuteRPG_Village/32x32/CuteRPG_Village_House.png",
    #     "tileSize": [32, 32],
    #     "transparentColor": [0, 0, 0],
    #     "sprites":
    #         {
    #             "house_corner_ul": { "baseTile": [4, 4],  "size": [1, 1] },
    #             "test1": { "baseTile": [4, 4],  "size": [1, 1] }
    #         }
    # }

    def loadSpritesheet(self, spritesheetData):
        sheetName = spritesheetData["sheetName"]
        filename = spritesheetData["filename"]
        tileSize = spritesheetData["tileSize"]
        offset = [0, 0]
        if "offset" in spritesheetData:
            offset = spritesheetData["offset"]
        sprites = spritesheetData["sprites"]
        composites = spritesheetData["composites"]

        # Load the spritesheet.  (convert_alpha() is used to load the alpha channel for each pixel, to handle transparency)
        spritesheet = pygame.image.load(self.assetPath + "/" + filename).convert_alpha()

        # For each sprite in the spritesheet, add it to the library
        count = 0
        for spriteName in sprites:
            sprite = sprites[spriteName]
            # Sprite starting location
            spriteX = sprite["baseTile"][0]
            spriteY = sprite["baseTile"][1]
            # Sprite size
            spriteWidthInTiles = sprite["size"][0]
            spriteHeightInTiles = sprite["size"][1]
            # Sprite tint color
            spriteColor = sprite.get("color", None)
            spriteAngle = sprite.get("angle", 0)

            # Get the sprite from the spritesheet
            #spriteRect = pygame.Rect(spriteX * tileSize[0], spriteY * tileSize[1], tileSize[0], tileSize[1])
            spriteRect = pygame.Rect((spriteX * tileSize[0]) + offset[0], (spriteY * tileSize[1]) + offset[1], spriteWidthInTiles * tileSize[0], spriteHeightInTiles * tileSize[1])
            spriteImage = spritesheet.subsurface(spriteRect)

            # If the sprite is less than 32x32, then add blank canvas to the right and bottom of the sprite
            # This SHOULD NOT SCALE THE SPRITE! It should only add blank canvas to the right and bottom of the sprite.
            if spriteImage.get_width() < 32:
                # Create a new surface
                newSpriteImage = pygame.Surface((32, spriteImage.get_height()), pygame.SRCALPHA, 32)
                # Blit the original sprite onto the new surface
                newSpriteImage.blit(spriteImage, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                # Set the new surface as the sprite
                spriteImage = newSpriteImage
            if spriteImage.get_height() < 32:
                # Create a new surface
                newSpriteImage = pygame.Surface((spriteImage.get_width(), 32), pygame.SRCALPHA, 32)
                # Blit the original sprite onto the new surface
                newSpriteImage.blit(spriteImage, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                # Set the new surface as the sprite
                spriteImage = newSpriteImage

            if spriteColor is not None:
                # Tint the sprite
                colorImage = pygame.Surface(spriteImage.get_size()).convert_alpha()
                colorImage.fill(pygame.Color(spriteColor))
                colorImage.blit(spriteImage, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                spriteImage = colorImage

            if spriteAngle:
                spriteImage = pygame.transform.rotate(spriteImage, spriteAngle)

            # Debug: Print the color at pixel 0,0 within the sprite
            #print("Color at 1,1: " + str(spriteImage.get_at((1,1))))

            # Add the sprite to the library
            # First, check for duplicate names
            if sheetName + "_" + spriteName in self.sprites:
                self.warnings.append("WARNING: Duplicate sprite name: " + sheetName + "_" + spriteName + ". Sprite will be overwritten.")
            self.sprites[sheetName + "_" + spriteName] = spriteImage

            count += 1

        # For each composite in the spritesheet, add it to the library
        # Composite format:
        # "composites": {
        #     "wall1": {"wall_horiz": [0, 0], "wall_t": [0, 1]}
        # }
        for compositeName in composites:
            composite = composites[compositeName]
            # Create a new surface for the composite
            compositeWidth = 0
            compositeHeight = 0
            for spriteName in composite:
                sprite = composite[spriteName]
                spriteX = sprite[0] * tileSize[0]
                spriteY = sprite[1] * tileSize[1]
                spriteImage = self.sprites[sheetName + "_" + spriteName]
                spriteWidth = spriteImage.get_width()
                spriteHeight = spriteImage.get_height()
                if spriteX + spriteWidth > compositeWidth:
                    compositeWidth = spriteX + spriteWidth
                if spriteY + spriteHeight > compositeHeight:
                    compositeHeight = spriteY + spriteHeight

            compositeSurface = pygame.Surface((compositeWidth, compositeHeight))

            # Add the sprites to the composite
            for spriteName in composite:
                sprite = composite[spriteName]
                spriteX = sprite[0] * tileSize[0]
                spriteY = sprite[1] * tileSize[1]
                spriteImage = self.sprites[sheetName + "_" + spriteName]
                compositeSurface.blit(spriteImage, (spriteX, spriteY))

            # Add the composite to the library
            # First, check for duplicate names
            if sheetName + "_" + compositeName in self.sprites:
                self.warnings.append("WARNING: Duplicate sprite name: " + sheetName + "_" + compositeName + ". Sprite will be overwritten.")
            self.sprites[sheetName + "_" + compositeName] = compositeSurface

            count += 1


        print("\tLoaded " + str(count) + " sprites from " + filename + ".")
        return count

    # Get a list of all sprite names in the library (sorted)
    def getSpriteNames(self):
        return sorted(self.sprites.keys())

    # Display a sprite
    # spriteName: The name of the sprite to display
    # window: The window to display the sprite in
    # x: The x coordinate of the top left corner of the sprite
    # y: The y coordinate of the top left corner of the sprite
    def renderSprite(self, window, spriteName, x, y, scale=1, adjustY=True):
        # Tile size
        tileSize = 32
        if (scale != 1):
            tileSize = int(32 * scale)

        if (spriteName not in self.sprites):
            print("WARNING: Sprite not found: " + str(spriteName))
            missingSpriteName = "missing_missing"
            if (missingSpriteName not in self.sprites):
                return
            spriteName = missingSpriteName
            #exit(1)
            #return
        sprite = self.sprites[spriteName]

        # Adjust the y-coordinate based on the sprite's height
        adjusted_y = y
        if (adjustY):
            adjusted_y = y - (sprite.get_height()*scale) + tileSize
            #offsetY = 0
            #if (scale != 1):
            #    offsetY = tileSize - (tileSize*scale)
            #adjusted_y -= offsetY
            #print("Original Y: " + str(y) + "    Adjusted y: " + str(adjusted_y))

        if (scale != 1):
            # Scale around the center of the sprite
            sprite = pygame.transform.scale(sprite, (int(sprite.get_width() * scale), int(sprite.get_height() * scale)))

        window.blit(sprite, (x, adjusted_y))
