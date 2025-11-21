# UtilSpriteSheetGrid.py
import argparse
from os.path import join as pjoin
import time

# Font
import pygame

from discoveryworld.constants import ASSETS_PATH


# DisplayGrid
# This function loads a spritesheet, and displays them all in a grid. It shows the grid coordinates for each sprite.
# This is useful for figuring out the coordinates of sprites in a spritesheet.
# Offset is an (x, y) tuple that is added to the coordinates of each sprite.
def displayGrid(filename, tileSize, offset, transparentColor, window, scale=1.0):
    pygame.font.init()

    # Load the spritesheet
    spritesheet = pygame.image.load(filename).convert()
    spritesheet.set_colorkey(transparentColor)

    # Display the spritesheet
    spritesheet = pygame.transform.scale(spritesheet, (int(spritesheet.get_width() * scale), int(spritesheet.get_height() * scale)))
    window.blit(spritesheet, offset)

    # Display the grid
    for y in range(0, spritesheet.get_height(), tileSize[1]):
    #for y in range(offset[1], spritesheet.get_height() + offset[1], tileSize[1]):
        pygame.draw.line(window, (255, 255, 255), (0 + offset[0], y + offset[1]), (spritesheet.get_width(), y + offset[1]))
        # pygame.draw.line(window, (255, 255, 255), (0, y), (spritesheet.get_width(), y))
    for x in range(0, spritesheet.get_width(), tileSize[0]):
    # for x in range(offset[0], spritesheet.get_width() + offset[0], tileSize[0]):
        pygame.draw.line(window, (255, 255, 255), (x + offset[0], 0 + offset[1]), (x + offset[0], spritesheet.get_height()))
        # pygame.draw.line(window, (255, 255, 255), (x, 0), (x, spritesheet.get_height()))

    # Display the grid coordinates
    font = pygame.font.SysFont("Arial", int(8*scale))
    for y in range(0, spritesheet.get_height(), tileSize[1]):
        for x in range(0, spritesheet.get_width(), tileSize[0]):
            text = font.render(str(x // tileSize[0]) + "," + str(y // tileSize[1]), True, (255, 255, 255))
            #text = pygame.transform.scale(text, (int(text.get_width() * scale), int(text.get_height() * scale)))
            window.blit(text, (x + 2 + offset[0], y + 2 + offset[1]))


def main(args):
    #filename = "assets/pixymoon/CuteRPG_Interior/32x32/CuteRPG_Interior_custom.png"
    # filename = "assets/pixymoon/CuteRPG_Caves/32x32/CuteRPG_Caves.png"
    #filename = "assets/pixymoon/CuteRPG_Forest/32x32/CuteRPG_Forest.png"
    #filename = "assets/pixymoon/CuteRPG_Forest/32x32/CuteRPG_Forest_modified2.png"
    #filename = "assets/pixymoon/CuteRPG_Houses/32x32/CuteRPG_Houses_A-modified.png"
    #filename = "assets/pixymoon/CuteRPG_Houses/32x32/CuteRPG_Houses_A-modified1.png"
    #filename = "assets/pixymoon/CuteRPG_Village/32x32/CuteRPG_Village.png"

    #filename = "assets/pixymoon/CuteRPG_Sprites/Characters/Character_018.png"

    tileSize = (32, 32)
    #tileSize = (24, 24)
    #offset = (8, 0)
    offset = (0, 0)
    scale = 1.0
    transparentColor = (0, 0, 0)

    windowSize = (1800, 1200)

    window = pygame.display.set_mode(windowSize)
    pygame.display.set_caption("Sprite Sheet Grid")
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(10)

        # Check for a quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the window with black
        window.fill((0, 0, 0))

        # Insert drawing code here
        # This is just a test that puts a red square in the middle of the screen
        #pygame.draw.rect(window, (255, 0, 0), (gameParams["width"] / 2 - 25, gameParams["height"] / 2 - 25, 50, 50))
        pygame.draw.rect(window, (255, 0, 0), (windowSize[0] / 2 - 25, windowSize[1] / 2 - 25, 50, 50))

        # Display the sprite sheet
        try:
            displayGrid(args.filename, tileSize, offset, transparentColor, window, scale)
        except:
            displayGrid(pjoin(ASSETS_PATH, args.filename), tileSize, offset, transparentColor, window, scale)

        # Flip the backbuffer
        pygame.display.flip()

        # Add zooming with x and z keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_x]:
            scale *= 2
            tileSize = (tileSize[0] * 2, tileSize[1] * 2)
        if keys[pygame.K_z]:
            scale //= 2
            tileSize = (tileSize[0] // 2, tileSize[1] // 2)

        # Use arrows to move center
        if keys[pygame.K_LEFT]:
            offset = (offset[0] + tileSize[0], offset[1])
            time.sleep(0.3)

        if keys[pygame.K_RIGHT]:
            offset = (offset[0] - tileSize[0], offset[1])
            time.sleep(0.3)

        if keys[pygame.K_UP]:
            offset = (offset[0], offset[1] + tileSize[1])
            time.sleep(0.3)

        if keys[pygame.K_DOWN]:
            offset = (offset[0], offset[1] - tileSize[1])
            time.sleep(0.3)



# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View sprite grid.")
    parser.add_argument('filename')

    args = parser.parse_args()

    # Main function
    main(args)