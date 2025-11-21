
import argparse
import shutil
import pygame
import os
import json
import time
import psutil
import textwrap
import random
import math
import sys

from os.path import join as pjoin

from discoveryworld.ActionSuccess import MessageImportance
from discoveryworld.KnowledgeScorer import Measurement
from discoveryworld.UserInterface import UserInterface
from discoveryworld.ScenarioMaker import ScenarioMaker, SCENARIOS, SCENARIO_NAMES, SCENARIO_INFOS, SCENARIO_DIFFICULTY_OPTIONS, getInternalScenarioName

from discoveryworld.World import World
from discoveryworld.constants import ASSETS_PATH


# Helper for showing a dialog box containing a number of options, with the user able to pick one using arrow keys/return.
def dialogPickOption(window, options:list, displayMessage:str=None):
    # Initialize Pygame fonts
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 15)
    fontBold = pygame.font.SysFont("monospace", 15, bold=True)

    # Main loop
    running = True
    currentOption = 0
    arrowKeyDown = False
    while running:
        # Clear the event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if (event.type == pygame.KEYDOWN):
                # Use arrow keys to select options (up/down).  Should move once when key is pressed, then reset when key is released
                if (event.key == pygame.K_UP):
                    if (arrowKeyDown == False):
                        currentOption -= 1
                        if (currentOption < 0):
                            currentOption = len(options) - 1
                        arrowKeyDown = True
                if (event.key == pygame.K_DOWN):
                    if (arrowKeyDown == False):
                        currentOption += 1
                        if (currentOption >= len(options)):
                            currentOption = 0
                        arrowKeyDown = True

                # Pressing RETURN will select the current option
                if (event.key == pygame.K_RETURN):
                    return options[currentOption]

                # Pressing ESC will quit the dialog, returning NONE
                if (event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE):
                    return None

            if (event.type == pygame.KEYUP):
                arrowKeyDown = False


        # Clear screen (optional, depends on design)
        window.fill((0, 0, 0))

        # Draw a starfield
        mkStarfield(window)
        # Show DiscoveryWorld Logo at the top
        # For now, just draw this in a large font

        # Display the DiscoveryWorld logo
        # Load the logo image
        logoImage = pygame.image.load(pjoin(ASSETS_PATH, "logo", "logo.png"))
        logoRect = logoImage.get_rect()
        logoRect.center = (window.get_width() // 2, 120)
        window.blit(logoImage, logoRect)

        # Display text message, centered, at 150 pixels down from the top of the screen
        if (displayMessage != None):
            textMessage = fontBold.render(displayMessage, True, (200, 200, 200))
            textRect = textMessage.get_rect()
            textRect.center = (window.get_width() // 2, 250)
            window.blit(textMessage, textRect)

        # Also display a message at the bottom to indicate how to select an option
        textMessage = fontBold.render("Use the arrow keys to scroll between options, and press RETURN to select an option.", True, (200, 200, 200))
        textRect = textMessage.get_rect()
        textRect.center = (window.get_width() // 2, window.get_height() - 50)
        window.blit(textMessage, textRect)

        # Display the options
        startOptionIdx = 0
        maxOptionsToShow = 9
        if (len(options) > maxOptionsToShow):
            startOptionIdx = currentOption - (maxOptionsToShow // 2)
            if (startOptionIdx < 0):
                startOptionIdx = 0
            if (startOptionIdx > len(options) - maxOptionsToShow):
                startOptionIdx = len(options) - maxOptionsToShow

        optionsShown = 0
        for i, optionStr in enumerate(options):
            # Background -- grey if not current option, blue if current option
            # All options should be centered

            if (i < startOptionIdx) or (i >= startOptionIdx + maxOptionsToShow):
                continue

            textOption = fontBold.render(optionStr, True, (200, 200, 200))
            textRect = textOption.get_rect()
            textRect.center = (window.get_width() // 2, 300 + (optionsShown * 40))

            # Draw the background rectangle
            # This should be a new rectangle, that's at least 300 pixels wide, and centered on the same center
            backgroundRect = pygame.Rect(textRect.left - 5, textRect.top - 5, textRect.width + 10, textRect.height + 10)
            # Make at least 300 pixels wide
            if (backgroundRect.width < 300):
                backgroundRect.width = 300
                backgroundRect.left = textRect.center[0] - 150

            if (i == currentOption):
                pygame.draw.rect(window, (0, 0, 200), (backgroundRect.left - 5, backgroundRect.top - 5, backgroundRect.width + 10, backgroundRect.height + 10))
            else:
                pygame.draw.rect(window, (100, 100, 100), (backgroundRect.left - 5, backgroundRect.top - 5, backgroundRect.width + 10, backgroundRect.height + 10))

            # Draw the background text, centered
            window.blit(textOption, textRect)

            optionsShown += 1


        # Update the display
        pygame.display.flip()

    return None


# Starfield animation
starfieldNumStars = 100
starfieldStarCoordinates = []
starfieldStarVelocities = []
starfieldStarAngles = []

def mkStarfield(window):
    # If the starfield has not been initialized, then initialize it
    if (len(starfieldStarCoordinates) == 0):
        for i in range(starfieldNumStars):
            x = random.randint(0, window.get_width())
            y = random.randint(0, window.get_height())
            starfieldStarCoordinates.append((x, y))
            # Velocity, in pixels per update
            randVelocity = random.uniform(0.1, 0.3)
            starfieldStarVelocities.append(randVelocity)
            starfieldStarAngles.append(random.randint(0, 360))

    # Draw the starfield
    for i in range(starfieldNumStars):
        x = starfieldStarCoordinates[i][0]
        y = starfieldStarCoordinates[i][1]
        # Draw the star
        pygame.draw.circle(window, (255, 255, 255), (x, y), 1)

        # Move the star
        angle = starfieldStarAngles[i]
        velocity = starfieldStarVelocities[i]
        x += velocity * math.cos(math.radians(angle))
        y += velocity * math.sin(math.radians(angle))

        # If the star goes off the edge of the screen, respawn it at the center of the screen
        if (x < 0) or (x > window.get_width()) or (y < 0) or (y > window.get_height()):
            x = window.get_width() // 2
            y = window.get_height() // 2
            angle = random.randint(0, 360)

        starfieldStarCoordinates[i] = (x, y)


# Main function for picking a scenario, difficulty, and task variation
def pickScenario(window):
    stage = "scenario"
    while stage != "done":
        # Choice 1: Select a scenario
        if stage == "scenario":
            choiceTaskName = dialogPickOption(window, SCENARIO_NAMES, displayMessage="Select a scenario:")
            if choiceTaskName is None:
                print("User quit the game.")
                pygame.quit()
                return

            stage = "difficulty"

        # Choice 2: Select a difficulty
        if stage == "difficulty":
            optionsDifficulty = SCENARIO_INFOS[choiceTaskName]["difficulty"]
            choiceDifficulty = dialogPickOption(window, optionsDifficulty, displayMessage="Select a difficulty:")
            if choiceDifficulty is None:
                stage = "scenario"
                continue  # Go back to the scenario selection

            stage = "variation"

        # Choice 3: Select a task variation
        if stage == "variation":
            optionsVariation = SCENARIO_INFOS[choiceTaskName]["variations"]
            choiceVariation = dialogPickOption(window, optionsVariation, displayMessage="Select a task variation:")
            if choiceVariation is None:
                stage = "difficulty"
                continue  # Go back to the difficulty selection

            # If we reach this point, then we have selected a scenario, difficulty, and task variation
            stage = "done"

    # Map between the choice and the scenario name
    scenarioName = getInternalScenarioName(choiceTaskName, choiceDifficulty)

    # Map between variation and random seed
    seed = int(choiceVariation)-1

    return {
        "scenario": scenarioName,
        "seed": seed
    }


# Save log
def saveLog(world, logInfo, verboseLogFilename, pygameWindow, pygame, lastScreenExportFilename):
    # Save the log file
    world.exportWorldHistoryJSON(logInfo, verboseLogFilename, pygameWindow, pygame, lastScreenExportFilename)
    print("Log file saved to: " + verboseLogFilename)
    # Archive the entire log directory
    archiveFilename = pjoin("logs", logInfo["verboseLogFilenameNoPathNoExt"] + ".zip")
    print("Archiving log directory to: " + archiveFilename)
    import zipfile
    with zipfile.ZipFile(archiveFilename, 'w') as zipf:
        for root, dirs, files in os.walk(logInfo["verboseFolder"]):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(logInfo["verboseFolder"], '..')))
    print("Log directory archived to: " + archiveFilename)




# Main entry point
def main(args):
    print("Initializing...")
    displayGrid = False

    # 32 pixels/tile * 32 tiles = 1024 pixels

    windowMode = "small"
    #windowMode = "big"

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
    lastScreenExportFilename = None
    if os.environ.get("MARC", False):
        scale = 2.0
        viewportSizeX //= 2
        viewportSizeY //= 2


    # Open window
    #window = pygame.display.set_mode((gameParams["width"], gameParams["height"]))
    # Open window (should open in top left corner)
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50,50)
    window = pygame.display.set_mode((gameParams["width"], gameParams["height"]))
    pygame.display.set_caption(gameParams["name"])

    clock = pygame.time.Clock()

    # Initialize font renderer
    pygame.font.init()

    # Show the screen to pick the scenario
    # If no scenario specified on the command line, then show the scenario picker
    if args.scenario is None:
        scenarioInfo = pickScenario(window)
        if scenarioInfo is None:
            return

        args.scenario = scenarioInfo["scenario"]
        args.seed = scenarioInfo["seed"]

    # Intialize world
    world = World(assetPath=None, filenameSpriteIndex="spriteIndex.json", dataPath=None, filenameObjectData="objects.tsv", filenameMaterialData="materials.tsv", filenameDiscoveryFeed="discoveryFeed.json")

    # Set that this is a live user playing (not an agent)
    world.setLiveUserPlaying()

    print ("All sprite names: ")
    print (world.spriteLibrary.getSpriteNames())

    # Initialize the user interface
    ui = UserInterface(window, world.spriteLibrary, showScoreToUser=args.showScoreToUser)

    # Create a new random number generator (for deterministic behavior) with a specific seed
    scenarioMaker = ScenarioMaker(world, seed=args.seed)
    smSuccess, smErrorStr = scenarioMaker.setupScenario(args.scenario)
    if (not smSuccess):
        print("ERROR: ScenarioMaker failed to setup scenario: " + smErrorStr)
        sys.exit(1)

    # Initial world tick
    world.tick()

    # Find a user agent
    userAgents = world.getUserAgents()
    if (len(userAgents) == 0):
        print("ERROR: No user agents found!")
        sys.exit(1)
    currentAgent = userAgents[0]

    # Attach the user interface to the agent
    ui.setAgent(currentAgent)

    # Set the initial task message for the user
    if (len(world.taskScorer.tasks) > 0):
        task = world.taskScorer.tasks[0]
        taskDescription = task.taskDescription

        welcomeStr = "-= Welcome to DiscoveryWorld! =-\n\n"
        welcomeStr += taskDescription + "\n\n"
        welcomeStr += "You are strongly encouraged to use external tools (notebooks, spreadsheets, statistics, etc.) to help you solve the task, as making these discoveries will require methodologically exploring different hypotheses. "
        welcomeStr += "Once the task is completed, the game will automatically end. "
        welcomeStr += "While playing, press ? or F1 for help, and TAB to display this task information again.\n\n"
        welcomeStr += "Press SPACE to close this message."

        # Add the task information to the text message queue
        ui.addTextMessageToQueue(welcomeStr)


    # Check that there's a "logs" subdirectory on the current folder
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)

    verboseFolderName = "discoveryworld-playlog-" + args.scenario.replace(" ", "_") + ".seed" + str(args.seed) + "." + time.strftime("%Y%m%d_%H%M%S")
    verboseFolder = pjoin(LOGS_DIR, verboseFolderName)
    os.makedirs(verboseFolder, exist_ok=True)

    # Also make a 'frames' directory
    verboseFramesFolder = pjoin(verboseFolder, "frames")
    os.makedirs(verboseFramesFolder, exist_ok=True)

    # Create a structure that records the log file parameters
    verboseLogFilenameNoPathNoExt = "discoveryworld-playlog-" + args.scenario.replace(" ", "_") + ".seed" + str(args.seed) + "." + time.strftime("%Y%m%d_%H%M%S") + ".log"
    verboseLogFilename = pjoin(verboseFolder, verboseLogFilenameNoPathNoExt + ".json")
    logInfo = {
        "scenario": args.scenario,
        "seed": args.seed,
        "scenarioName": args.scenario,
        "taskDescription": taskDescription,
        # Add the date and time started
        "dateStarted": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Make a verbose filename for the log
        "verboseFolder": verboseFolder,
        "verboseLogFilename": verboseLogFilename,
        "verboseLogFilenameNoPathNoExt": verboseLogFilenameNoPathNoExt,
    }

    # Main rendering loop
    running = True
    frames = 0
    autoRunCycles = 0
    lastMove = time.time()        # Time of last move (in seconds since start of game)
    lastSize = 0
    taskCompletedMessageShown = False
    confirmingQuit = False
    saveNextFrame = False
    saveSuccessful = False
    while running:
        #print("Frame: " + str(frames))
        exportFrame = False

        # Check for a request to save the frame
        if (saveNextFrame) and (not ui.inModal):
            saveLog(world, logInfo, verboseLogFilename, pygameWindow=window, pygame=pygame, lastScreenExportFilename=lastScreenExportFilename)
            saveNextFrame = False
            saveSuccessful = True

        curTime = time.time()
        clock.tick(gameParams["fps"])

        # Check for a quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check whether the task has been completed -- and if so, show a message (AND, export the logfile)
        tasks = world.taskScorer.tasks
        task = None
        if (len(tasks) > 0):
            task = tasks[0]
            if (task.isCompleted() == True) and (taskCompletedMessageShown == False):
                # If we're here, the task is completed.  Now check if it's an appropriate time to mark it as completed
                if (not ui.inModal) and (ui.dialogToDisplay == None):    # Make sure we're not in a modal dialog, or actively engaged in dialog
                    taskCompletedMessageShown = True
                    taskCompletedMessage = "THE GAME HAS ENDED.\n\n"
                    if (task.isCompletedSuccessfully() == True):
                        taskCompletedMessage += "Congratulations! You have completed the task successfully.\n"
                    else:
                        taskCompletedMessage += "Unfortunately, the task was not completed successfully.\n"

                    taskCompletedMessage += "\n"
                    taskCompletedMessage += "Task Description:\n"
                    taskCompletedMessage += taskDescription + "\n\n"

                    # taskScore = int(task.getScoreNormalized() * 100)
                    # taskCompletedMessage += "Task Score: " + str(taskScore) + "%\n"
                    # taskCompletedMessage += "\n"
                    taskCompletedMessage += "Press SPACE to close this message, then press ESC to quit the game (or F5 to keep playing)."

                    ui.addTextMessageToQueue(taskCompletedMessage)

                    # Export the logfile
                    saveNextFrame = True
                    #saveLog(world, logInfo, verboseLogFilename, pygameWindow=window, pygame=pygame, lastScreenExportFilename=lastScreenExportFilename)

        # Check for keyboard input
        keys = pygame.key.get_pressed()

        # Signify whether the agent has done their move this turn
        doNextTurn = False
        if (ui.inModal):

            # If a 'quit confirm' modal is open, then pressing 'Y' will quit the game
            if (confirmingQuit == True):
                # Quit modal
                if (keys[pygame.K_y]):
                    running = False
                if (keys[pygame.K_n]):
                    ui.closeModal()
                    confirmingQuit = False
                    time.sleep(0.50)     # wait for half a second for the key to be released
                    # Flush the key buffer
                    pygame.event.clear()

            elif (ui.inDiscoveryFeedModal):
                print("#### IN DISCOVERY FEED MODAL")
                # Discovery Feed modal: Pressing SPACE or RETURN will close the modal
                if (keys[pygame.K_SPACE] or keys[pygame.K_RETURN] or keys[pygame.K_ESCAPE]):
                    ui.closeModal()
                    ui.inDiscoveryFeedModal = False
                    doNextTurn = True
                elif (keys[pygame.K_PAGEUP]):
                    ui.closeModal()                                         # Close the current discoveryfeed view
                    (doTick, success) = ui.discoveryFeedViewDecrement()     # And add on a new one
                    time.sleep(0.2)
                    # Update the UI with the message (for the bottom of the screen)
                    ui.updateLastActionMessage(success.message)
                    # If the message was rated as high importance, also add it to the explicit modal dialog queue
                    if (success.importance == MessageImportance.HIGH):
                        ui.addTextMessageToQueue(success.message)

                elif (keys[pygame.K_PAGEDOWN]):
                    ui.closeModal()                                         # Close the current discoveryfeed view
                    (doTick, success) = ui.discoveryFeedViewIncrement()     # And add on a new one
                    time.sleep(0.2)
                    # Update the UI with the message (for the bottom of the screen)
                    ui.updateLastActionMessage(success.message)
                    # If the message was rated as high importance, also add it to the explicit modal dialog queue
                    if (success.importance == MessageImportance.HIGH):
                        ui.addTextMessageToQueue(success.message)

            else:
                # General modal: Pressing SPACE or RETURN will close the modal
                if (keys[pygame.K_SPACE] or keys[pygame.K_RETURN] or keys[pygame.K_ESCAPE]):
                    ui.closeModal()
                    doNextTurn = True

        else:
            # Parse any action keys
            doTick = True
            success = None
            if (taskCompletedMessageShown == False) or (ui.extendedPlayEnabled == True):
                # Only parse action keys if the task has not been completed
                (doTick, success) = ui.parseKeys(keys)

            if (success != None):
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
                    ui.addTextMessageToQueue("Are you sure you want to quit the game?\nY - Quit\nN - Continue")
                    confirmingQuit = True

                if (keys[pygame.K_w]):
                    # Wait a turn
                    print("Waiting (taking no action this turn)...")
                    doNextTurn = True

                if (keys[pygame.K_F10]):
                    # Save a full-screen screenshot of the game
                    # Filename should be screenshot.<taskname>.<seed>.<timestamp>.png
                    screenshotFilename = "screenshot." + args.scenario.replace(" ", "_") + ".seed" + str(args.seed) + "." + time.strftime("%Y%m%d_%H%M%S") + ".png"
                    # Re-render the world, including the entire world
                    # def renderViewport(self, window, worldStartX, worldStartY, sizeTilesX, sizeTilesY, offsetX, offsetY, scale=1.0, includeGrid=False):
                    #world.renderViewport(window, 0, 0, world.width, world.height, 0, 0, scale=scale, includeGrid=displayGrid)
                    # Make a faux window that's the size of the entire world
                    fauxWindow = pygame.Surface((world.sizeX * 32, world.sizeY * 32))
                    # Render the world to the faux window
                    world.renderViewport(fauxWindow, 0, 0, world.sizeX, world.sizeY, 0, 0, scale=scale, includeGrid=displayGrid)
                    # Save the screenshot
                    pygame.image.save(fauxWindow, screenshotFilename)
                    print("Screenshot saved to: " + screenshotFilename)
                    # Close the faux window
                    del fauxWindow
                    # Sleep for 0.5 seconds to prevent accidental double-press
                    time.sleep(0.5)


                # F5 to continue playing
                elif (keys[pygame.K_F5]) and (taskCompletedMessageShown == True):
                    print("Extended play enabled.")

                    # Check to make sure the task was completed successfully
                    if (not task.isCompletedSuccessfully() == True):
                        extendedPlayMessage = "Unfortunately, extended play mode is only available when the task is completed successfully, "
                        extendedPlayMessage += "in case you need more time to build an explanation for the solution.\n\n"
                        extendedPlayMessage += "Press SPACE to close this message."
                        ui.addTextMessageToQueue(extendedPlayMessage)

                    # Check to make sure the scenario isn't `tutorial`
                    elif (args.scenario == "tutorial"):
                        # Extended play not supported for the tutorial
                        extendedPlayMessage = "Unfortunately, extended play mode is not available for the tutorial scenario.\n\n"
                        extendedPlayMessage += "Press SPACE to close this message."
                        ui.addTextMessageToQueue(extendedPlayMessage)

                    else:
                        ui.extendedPlayEnabled = True

                        extendedPlayMessage = "Extended play mode enabled.  You can now play as normal.\n\n"
                        extendedPlayMessage += "When you're done, press ESC to quit the game.\n"
                        extendedPlayMessage += "(The game will automatically save the log file when you quit.)\n\n"
                        extendedPlayMessage += "Press SPACE to close this message."
                        ui.addTextMessageToQueue(extendedPlayMessage)


                # Test the text message queue
                elif (keys[pygame.K_TAB]):
                    # Test the text message queue

                    #ui.addTextMessageToQueue("This is a test\nThis is an extra long line apple orange banana pineapple fruit flower tree coconut sky water air summer water blue green yellow orange purple this is the end of the second line.\nThis is the second line.")
                    #tasks = world.taskScorer.tasks
                    if (task is None):
                        ui.addTextMessageToQueue("Task Information:\n\nNo tasks are currently active.\n\nPress SPACE to close this message.")
                    else:
                        taskDescription = task.taskDescription
                        taskScore = int(task.getScoreNormalized() * 100)
                        isCompleted = task.isCompleted()
                        isCompletedSuccessfully = task.isCompletedSuccessfully()

                        taskStr = "Task Description:\n\n"

                        taskStr += taskDescription + "\n\n"
                        taskStr += "You are welcome to use external tools (notebooks, spreadsheets, statistics, etc.) to help you solve the task. "
                        taskStr += "Once the task is completed, the game will automatically end.\n\n"

                        #taskStr += "Task Score: " + str(taskScore) + "%\n"
                        # taskStr += "Task Completed: " + str(isCompleted) + "\n"
                        # taskStr += "Task Completed Successfully: " + str(isCompletedSuccessfully) + "\n\n"
                        taskStr += "Press SPACE to close this message."

                        # Add the task information to the text message queue
                        ui.addTextMessageToQueue(taskStr)

                # Help Screen (question mark/slash key)
                elif keys[pygame.K_SLASH] or keys[pygame.K_QUESTION] or keys[233] or keys[pygame.K_F1]:
                    # Display the help screen
                    helpStr = "DiscoveryWorld Help\n\n"
                    helpStr += "Arrow keys: Move the agent\n"
                    helpStr += "TAB:   View current task information\n"
                    helpStr += "SPACE: Pick up object (in Arg 1 slot)\n"
                    helpStr += "D:     Drop inventory item (in Arg 1 slot)\n"
                    helpStr += "P:     Put an item (Arg 1) in a specific container (Arg 2)\n"
                    helpStr += "P:     Give an item (Arg 1) to another character (Arg 2)\n"
                    helpStr += "O/C:   Open/close a container (in Arg 1 slot)\n"
                    helpStr += "A/S:   Activate/deactive a device (in Arg 1 slot)\n"
                    helpStr += "T:     Talk to another character (in Arg 1 slot)\n"
                    helpStr += "E:     Eat an item (in Arg 1 slot)\n"
                    helpStr += "U:     Use an item (Arg 1), optionally on another item (Arg 2)\n"
                    helpStr += "R:     Read an item (in Arg 1 slot)\n"
                    helpStr += "Z:     Teleport to a random important location\n"
                    helpStr += "W:     Wait a turn (do nothing)\n"
                    helpStr += "V:     View recent posts on Discovery Feed\n"
                    helpStr += "0-9:   Select a specific inventory item (Arg 1), hold shift for (Arg2)\n"
                    helpStr += "[ / ]: Cycle through inventory items (Arg 1)\n"
                    helpStr += "; / \": Cycle through inventory items (Arg 2)\n"
                    helpStr += "         (Note: Arguments with a purple border are in the inventory,\n"
                    helpStr += "         arguments with a green border are in the world)\n"

                    helpStr += "?/F1:  Display this help message\n"
                    helpStr += "ESC:   Quit the game\n"
                    helpStr += "\n"
                    helpStr += "Example: To 'use the shovel on the soil':\n"
                    helpStr += " - Use [ and/or ] keys to select the shovel (arg 1)\n"
                    helpStr += " - Use ; and/or \" keys to select the soil (arg 2)\n"
                    helpStr += " - Press U to use the shovel on the soil.\n"
                    helpStr += "\n"
                    helpStr += "Press SPACE to close this message."

                    # Add the help message to the text message queue
                    ui.addTextMessageToQueue(helpStr)
                    doNextTurn = False

        # Rendering

        # Fill the window with black
        window.fill((0, 0, 0))

        # Update the world
        # If the agent has taken their turn, then update the world
        if (doNextTurn) or (autoRunCycles > 0):
            # Update the world

            # If the task is completed, then the world will not tick
            doUpdate = True
            if ((task is not None) and (ui.inModal == False) and (ui.dialogToDisplay == None) and ((task.isCompleted() == True) and (ui.extendedPlayEnabled == False))):
                doUpdate = False
            #if (task is None) or (task.isCompleted() == False):
            if (doUpdate):
                # Update the world
                world.tick()

            if args.debug:
                # Report task progress
                print(world.taskScorer.taskProgressStr())

            frames += 1
            print("\n\n############################################################################################")
            if (autoRunCycles > 0):
                print("Step: " + str(frames) + " (autorun)")
            else:
                print("Step: " + str(frames))

            # Print action
            if (success != None):
                print("Feedback: " + success.message)

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

            ## Debug: Export the logfile every 10 steps
            #if (frames % 10 == 0):
            #    saveLog(world, logInfo, verboseLogFilename, pygameWindow=window, pygame=pygame, lastScreenExportFilename=lastScreenExportFilename)

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

            print("############################################################################################\n")

            if (autoRunCycles > 0):
                autoRunCycles -= 1

            exportFrame = True


        # Render the world
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

        # # Find a usable item at the location the agent is facing
        # facingLocation = currentAgent.getWorldLocationAgentIsFacing()
        # # Bound checking
        # if (world.isWithinBounds(facingLocation[0], facingLocation[1])):
        #     # Get objects at location
        #     objs = world.getObjectsAt(facingLocation[0], facingLocation[1])
        #     # Step 6: Display the object inventory box
        #     ui.renderObjectSelectionBox(objs, curSelectedObjIdx=2)

        #     print("List of objects the agent is facing: " + str(objs))


        # Save the screen frame to a file
        if (exportFrame):
            frameFilename = pjoin(verboseFramesFolder, "frame_" + str(frames) + ".png")
            pygame.image.save(window, frameFilename)
            time.sleep(0.1)
            lastScreenExportFilename = frameFilename

        # Display the sprite
        #world.spriteLibrary.renderSprite(window, "house1_wall1", 100, 100)

        # Flip the backbuffer
        pygame.display.flip()
        #frames += 1

        #time.sleep(1)

    # If we get here, the game loop is over.
    # Convert the frames to a video
    # print("Converting frames to video...")
    # Call FFMPEG (forces overwrite)
    #subprocess.call(["ffmpeg", "-y", "-framerate", "10", "-i", FRAME_DIR + "/frame_%d.png", "-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-pix_fmt", "yuv420p", "output.mp4"])

    if (saveSuccessful == False) or (ui.extendedPlayEnabled == True):
        # Save the data, because the game is exiting but the data hasn't been saved yet (i.e. from having completed the task, which triggers a save)
        saveLog(world, logInfo, verboseLogFilename, pygameWindow=window, pygame=pygame, lastScreenExportFilename=lastScreenExportFilename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Play DiscoveryWorld.")
    parser.add_argument('--scenario', choices=SCENARIOS, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--showScoreToUser', action="store_true", default=False)

    args = parser.parse_args()

    print("Starting DiscoveryWorld...")
    print("Scenario: " + str(args.scenario))
    print("Seed: " + str(args.seed))

    main(args)
