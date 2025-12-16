
import argparse
import pygame
import os
import json
import random
import time
import subprocess
import psutil
from termcolor import colored
from discoveryworld.ActionSuccess import MessageImportance
from discoveryworld.KnowledgeScorer import Measurement
from discoveryworld.UserInterface import UserInterface
from discoveryworld.ScenarioMaker import ScenarioMaker, SCENARIOS

from discoveryworld.World import World

# Sprite library
# import SpriteLibrary
# from ObjectMaker import ObjectMaker
# from World import World
# from Layer import Layer
# from BuildingMaker import BuildingMaker
# from ScenarioMaker import *
# from ObjectModel import *
# from Agent import *
# from ActionSuccess import *
# from UserInterface import UserInterface
# from DialogTree import DialogMaker
# from KnowledgeScorer import *

# from JSONEncoder import CustomJSONEncoder


def main(args):
    print("Initializing...")
    displayGrid = False

    # 32 pixels/tile * 32 tiles = 1024 pixels

    #windowMode = "small"
    windowMode = "big"

    # Game parameters
    if windowMode == "small":
        gameParams = {
            "height": 750,
            "width": 800,
            "fps": 60,
            "name": "DiscoveryWorld"
        }

        # Step 2: Define the viewport size (in tiles)
        viewportSizeX = 24
        viewportSizeY = 16
    elif windowMode == "big":
        gameParams = {
            "height": 1024,
            "width": 1024,
            "fps": 30,
            "name": "DiscoveryWorld"
        }

        # Step 2: Define the viewport size (in tiles)
        viewportSizeX = 32
        viewportSizeY = 32

    scale = 1.0

    # Open window
    #window = pygame.display.set_mode((gameParams["width"], gameParams["height"]))
    # Open window (should open in top left corner)
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50,50)
    window = pygame.display.set_mode((gameParams["width"], gameParams["height"]))
    pygame.display.set_caption(gameParams["name"])

    clock = pygame.time.Clock()

    # Initialize font renderer
    pygame.font.init()

    # Intialize world
    world = World(assetPath=None, filenameSpriteIndex="spriteIndex.json", dataPath=None, filenameObjectData="objects.tsv", filenameMaterialData="materials.tsv", filenameDiscoveryFeed="discoveryFeed.json")
    print ("All sprite names: ")
    print (world.spriteLibrary.getSpriteNames())

    # Initialize the user interface
    ui = UserInterface(window, world.spriteLibrary, showScoreToUser=True)

    # Create a new random number generator (for deterministic behavior) with a specific seed
    #r = random.Random()

    scenarioMaker = ScenarioMaker(world, seed=args.seed)
    smSuccess, smErrorStr = scenarioMaker.setupScenario(args.scenario)
    if (not smSuccess):
        print("ERROR: ScenarioMaker failed to setup scenario: " + smErrorStr)
        exit(1)

    # Initial world tick
    world.tick()


    # Find a user agent
    userAgents = world.getUserAgents()
    if (len(userAgents) == 0):
        print("ERROR: No user agents found!")
        exit(1)
    currentAgent = userAgents[0]

    # Attach the user interface to the agent
    ui.setAgent(currentAgent)


    # Create a directory "/video" for storing video frames
    VIDEO_DIR = "video"
    FRAME_DIR = "video/frames"
    if (not os.path.exists(VIDEO_DIR)):
        os.mkdir(VIDEO_DIR)
    if (not os.path.exists(FRAME_DIR)):
        os.mkdir(FRAME_DIR)
    # Empty the frames directory
    for filename in os.listdir(FRAME_DIR):
        os.remove(FRAME_DIR + "/" + filename)


    # Main rendering loop
    running = True
    frames = 0
    autoRunCycles = 0
    lastMove = time.time()        # Time of last move (in seconds since start of game)
    lastSize = 0

    while running:
        #print("Frame: " + str(frames))
        exportFrame = False

        curTime = time.time()
        clock.tick(gameParams["fps"])

        # Check for a quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # if event.type == pygame.KEYDOWN:
            #     # print key
            #     print(colored(f"KEYDOWN: {event.key} | {event.unicode} | {pygame.key.name(event.key)}", 'red'))
            #     time.sleep(0.1)
            #     # Check for keyboard input
            #     keys = pygame.key.get_pressed()
            #     print(colored(keys.count(1), "red"))

        # Check for keyboard input
        keys = pygame.key.get_pressed()

        # Signify whether the agent has done their move this turn
        doNextTurn = False

        if (ui.inModal):
            # Pressing any key will close the modal
            if (keys[pygame.K_SPACE] or keys[pygame.K_RETURN]):
                ui.closeModal()
                doNextTurn = True

        else:

            # Parse any action keys
            (doTick, success) = ui.parseKeys(keys)

            if success is not None:
                # Export the frame, regardless of whether the action was successful, or whether it was a non-tick (e.g. UI) action
                exportFrame = True

                # Action was performed
                if (doTick):
                    doNextTurn = True

                # Update the UI with the message (for the bottom of the screen)
                ui.updateLastActionMessage(success.message)

                # If the message was rated as high importance, also add it to the explicit modal dialog queue
                if (success.importance == MessageImportance.HIGH):
                    ui.addTextMessageToQueue(success.message)

            else:

                # Escape -- quits the game
                if (keys[pygame.K_ESCAPE]):
                    running = False

                if (keys[pygame.K_g]):
                    displayGrid = not displayGrid
                    doNextTurn = True
                    success = True

                if (keys[pygame.K_x]):
                    # Zoom in
                    scale = (scale % 2) + 1.0
                    viewportSizeX = 32 // int(scale)
                    viewportSizeY = 32 // int(scale)
                    doNextTurn = False
                    time.sleep(0.3)

                # Manual "wait"
                elif (keys[pygame.K_w]):
                    # Wait a turn
                    print("Waiting (taking no action this turn)...")
                    doNextTurn = True

        # Rendering

        # Fill the window with black
        window.fill((0, 0, 0))

        # Update the world
        # If the agent has taken their turn, then update the world
        if (doNextTurn) or (autoRunCycles > 0):
            # Update the world
            world.tick()

            # Report task progress
            if args.debug:
                print(world.taskScorer.taskProgressStr())

            frames += 1
            print("\n\n############################################################################################")
            if (autoRunCycles > 0):
                print("Step: " + str(frames) + " (autorun)")
            else:
                print("Step: " + str(frames))

            if args.debug:
                # Print current memory usage
                curSize = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                delta = curSize - lastSize
                print("Current memory usage: " + str( curSize ) + " MB      (delta: " + str(delta) + " MB)")
                lastSize = curSize

            # Show all discovery feed posts
            print("Discovery feed:")
            for post in world.discoveryFeed.updatePosts:
                print(post)


            # Dump the world history to a JSON file
            # prevStep = world.step - 1
            # filenameOut = "history/worldHistory." + str(prevStep) + ".json"
            # with open(filenameOut, 'w') as outfile:
            #     # Get the last step of the world history (requires decompressing)
            #     lastHistoryStep= world.getWorldHistoryAtStep(world.step-1)
            #     if (lastHistoryStep != None):
            #         json.dump(lastHistoryStep, outfile, indent=4, cls=CustomJSONEncoder)

            ## Debug: dump the world history to a JSON file
            ## filenameWorldHistoryOut = "sandbox/worldHistory.pickle"
            ## world.exportWorldHistory(filenameWorldHistoryOut)

            # Test out the knowledge scorer
            print("--------------------")

            knowledgeScorer = currentAgent.knowledgeScorer
            # Create a measurement
            #         jsonStr = """
            # {
            #     "object": {"objectUUID": 1234, "scope":[{"propertyName":"color", "propertyOperator":"equals", "propertyValue":"red"}, {"propertyName":"size", "propertyOperator":"less_than", "propertyValue":5}]},
            #     "property": {"propertyName":"color", "propertyOperator":"equals", "propertyValue":"blue"}
            # }
            #     """
            jsonStr = """
                {
                    "object": {"objectType": "mushroom", "scope":[]},
                    "property": {"propertyName":"color", "propertyOperator":"equals", "propertyValue":"red"}
                }
            """

            # Convert to a dictionary
            dictIn = json.loads(jsonStr)
            # Create a Measurement
            measurement = Measurement(dictIn, step=world.step-1)
            print(measurement.errors)
            print(measurement)
            # Score the measurement
            score = knowledgeScorer.evaluateMeasurement(measurement)
            print("Score: " + str(score))
            print("Score justification: " + str(measurement.scoreJustification))

            print("############################################################################################\n")

            if (autoRunCycles > 0):
                autoRunCycles -= 1

            exportFrame = True

        # Render the world
        #world.render(window, cameraX=0, cameraY=0)

        # Render a viewport centered on the agent
        # def renderViewport(self, window, worldStartX, worldStartY, sizeTilesX, sizeTilesY, offsetX, offsetY):
        # Step 1: Get the agent's location
        agentLocation = currentAgent.getWorldLocation()

        # Step 3: Determine the worldStartX and worldStartY coordinates
        worldStartX = agentLocation[0] - int(viewportSizeX / 2)
        worldStartY = agentLocation[1] - int(viewportSizeY / 2)
        # Step 4: Render the viewport

        world.renderViewport(window, worldStartX, worldStartY, viewportSizeX, viewportSizeY, 0, 0, scale=scale, includeGrid=displayGrid)
        # Step 5: Render the user interface
        ui.render()

        # Save the screen frame to a file
        if (exportFrame):
            frameFilename = FRAME_DIR + "/frame_" + str(frames) + ".png"
            pygame.image.save(window, frameFilename)
            time.sleep(0.10)

        # Display the sprite
        #world.spriteLibrary.renderSprite(window, "house1_wall1", 100, 100)

        # Flip the backbuffer
        pygame.display.flip()

    # If we get here, the game loop is over.
    # Convert the frames to a video
    print("Converting frames to video...")
    # Call FFMPEG (forces overwrite)
    #subprocess.call(["ffmpeg", "-y", "-framerate", "10", "-i", FRAME_DIR + "/frame_%d.png", "-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-pix_fmt", "yuv420p", "output.mp4"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Play DiscoveryWorld.")
    parser.add_argument('--scenario', choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    print("Starting DiscoveryWorld...")
    print("Scenario: " + args.scenario)
    print("Seed: " + str(args.seed))

    main(args)
