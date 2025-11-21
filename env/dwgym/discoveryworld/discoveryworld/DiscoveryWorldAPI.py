# DiscoveryWorldAPI.py
import io
import os
import base64
import random
import subprocess

import pygame
from discoveryworld.ScenarioMaker import ScenarioMaker, SCENARIOS, SCENARIO_NAMES, SCENARIO_INFOS, SCENARIO_DIFFICULTY_OPTIONS, getInternalScenarioName
from discoveryworld.ActionHistory import getActionDescriptions
from discoveryworld.ActionSuccess import MessageImportance
from discoveryworld.World import World
from discoveryworld.UserInterface import UserInterface


#
#   DiscoveryWorld API
#
class DiscoveryWorldAPI:
    def __init__(self, threadID:int=1):
        self.NAME = "DiscoveryWorld API"
        self.VERSION = "0.1"
        self.THREAD_ID = threadID
        self.FRAME_DIR = "video/frames-thread" + str(threadID) + "/"

        #self.viewportSizeX = 15
        #self.viewportSizeY = 15

        #self.viewportSizeX = 9
        #self.viewportSizeY = 9
        #self.renderScale = 1.5

        #self.viewportSizeX = 7
        #self.viewportSizeY = 7
        #self.renderScale = 2.0

        # Step 2: Define the viewport size (in tiles)
        self.viewportSizeX = 24
        self.viewportSizeY = 16
        self.renderScale = 2.0


        # Create a random number generator
        self.r = random.Random()
        self.r.seed(0)

        # Create the window
        self.window = self.createHeadlessWindow()

        # World
        self.world = None

        # User interfaces (one per agent)
        self.numUserAgents = 0
        self.ui = []

        # Which agents have already performed actions this step
        self.agentsThatHaveActedThisStep = set()

        # Most recent task progress (updated from most recent observation)
        self.taskProgress = []

        # Most recent step count
        self.steps = 0


    def getNameAndVersion(self):
        return self.NAME + " v" + self.VERSION

    def getValidScenarios(self):
        return SCENARIO_INFOS

    def loadScenario(self, scenarioName:str, difficultyStr:str, randomSeed:int=0, numUserAgents = 1):
        # Set the number of agents
        self.numUserAgents = numUserAgents

        # Initialize a world (blank slate)
        self.world = World(assetPath=None, filenameSpriteIndex="spriteIndex.json", dataPath=None, filenameObjectData="objects.tsv", filenameMaterialData="materials.tsv", filenameDiscoveryFeed="discoveryFeed.json")

        # Get the internal name for the scenario
        internalScenarioName = getInternalScenarioName(scenarioName, difficultyStr)
        if (internalScenarioName == None):
            print("ERROR: Scenario not found: " + scenarioName)
            print(SCENARIO_INFOS)
            return False

        # Create a scenario
        scenarioMaker = ScenarioMaker(world=self.world, seed=randomSeed)
        success, errorStr = scenarioMaker.setupScenario(internalScenarioName, self.numUserAgents)

        if (not success):
            print("ERROR: " + errorStr)
            exit(1)
            return False

        # Clear the frame out directory
        # NOTE: If multiple agents are being run at the same time, we can't do this -- either need to keep a run-specific directory, or ask the user to provide a unique ID to use for the thread that we append to filenames to prevent collisions?
        if (os.path.exists(self.FRAME_DIR)):
            print("Removing existing frames from directory: " + self.FRAME_DIR)
            for filename in os.listdir(self.FRAME_DIR):
                if filename.endswith(".png"):
                    os.remove(os.path.join(self.FRAME_DIR, filename))

        # If the directory doesn't exist, create it
        if (not os.path.exists(self.FRAME_DIR)):
            os.makedirs(self.FRAME_DIR)

        # Initialize and attach user interfaces to the agents
        userAgents = self.world.getUserAgents()
        for userAgent in userAgents:
            ui = UserInterface(self.window, self.world.spriteLibrary)    # Create a user interface for this agent
            ui.setAgent(userAgent)                                  # Attach this UI to this agent
            self.ui.append(ui)                                      # Store it
        self.numUserAgents = len(userAgents)                        # If the number of user agents happens to be less for some reason (e.g. if the scenario has a maximum number it supports), then update the number of user agents

        # Initial world tick
        self.world.tick()

        # Reset task progress, since we're starting a new scenario
        self.taskProgress = []
        self.steps = 0

        return True




    # Gets the current observation of the world from a given agent's perspective
    def getAgentObservation(self, agentIdx):
        # Start populating response
        response = {"errors": [], "ui": {}, "vision": {}}

        # Check to make sure the agent index is valid
        if (agentIdx < 0) or (agentIdx >= self.numUserAgents):
            response["errors"].append("Agent index out of range. Specified agent index: " + str(agentIdx) + ". Number of agents: " + str(self.numUserAgents) + " (i.e. value must be between 0 and " + str(self.numUserAgents - 1) + ")")
            return response

        # Check that the world is initialized
        if (self.world == None):
            response["errors"].append("World is not initialized")
            return response


        # Get a reference to this agent's UI
        ui = self.ui[agentIdx]

        # Get a reference to the agent
        agent = ui.currentAgent

        # Get the agent's current location
        agentLocation = agent.getWorldLocation()

        # Define the viewport for this agent
        worldStartX = agentLocation[0] - int(self.viewportSizeX / 2)
        worldStartY = agentLocation[1] - int(self.viewportSizeY / 2)

        # Step 3: Render the viewport (the world view) and the UI for this agent
        self.window.fill((0, 0, 0)) # Clear the viewport
        self.world.renderViewport(self.window, worldStartX, worldStartY, self.viewportSizeX, self.viewportSizeY, 0, 0, self.renderScale, includeGrid=False)
        ui.render()                 # Render UI
        pygame.display.flip()       # Flip the backbuffer, to display this content to the window

        # Capture the current window, and save to file
        curStep = self.world.getStepCounter()
        filenameOutPNG = self.FRAME_DIR + "/ui_agent_" + str(agentIdx) + "_frame_" + str(curStep) + ".png"
        pygame.image.save(self.window, filenameOutPNG)

        #self.viewportSizeX = 24
        #self.viewportSizeY = 16
        #self.renderScale = 1.0

        agentVisionWidth = self.viewportSizeX * 32 * self.renderScale
        agentVisionHeight = self.viewportSizeY * 32 * self.renderScale

        # Also capture just the first 512x512 pixels of the window, and encode it as a base64 string
        # This is for the agent's "vision"
        visionSurface = pygame.Surface((agentVisionWidth, agentVisionHeight))
        visionSurface.blit(self.window, (0, 0), (0, 0, agentVisionWidth, agentVisionHeight))
        image_io = io.BytesIO()
        pygame.image.save(visionSurface, image_io, 'PNG')                       # Convert to PNG
        image_io.seek(0)                                                        # Go to the beginning of the BytesIO object
        encodedImageNoGrid = base64.b64encode(image_io.read()).decode('utf-8')       # Convert to base64
        encodedImageNoGrid = "data:image/png;base64," + encodedImageNoGrid
        response["vision"]["base64_no_grid"] = encodedImageNoGrid

        # Step 4: Also capture the viewport with the grid
        self.window.fill((0, 0, 0)) # Clear the viewport
        self.world.renderViewport(self.window, worldStartX, worldStartY, self.viewportSizeX, self.viewportSizeY, 0, 0, self.renderScale, includeGrid=True)
        ui.render()                 # Render UI
        pygame.display.flip()       # Flip the backbuffer, to display this content to the window

        # Capture the first 512x512 pixels of the window, and encode it as a base64 string
        # This is for the agent's "vision"
        visionSurface = pygame.Surface((agentVisionWidth, agentVisionHeight))
        visionSurface.blit(self.window, (0, 0), (0, 0, agentVisionWidth, agentVisionHeight))
        image_io = io.BytesIO()
        pygame.image.save(visionSurface, image_io, 'PNG')                       # Convert to PNG
        image_io.seek(0)                                                        # Go to the beginning of the BytesIO object
        encodedImageWithGrid = base64.b64encode(image_io.read()).decode('utf-8')       # Convert to base64
        encodedImageWithGrid = "data:image/png;base64," + encodedImageWithGrid
        response["vision"]["base64_with_grid"] = encodedImageWithGrid

        # Also dump this 'with grid' version to a debug file, called "current_viewport.png"
        pygame.image.save(visionSurface, self.FRAME_DIR + "/ui_agent_" + str(agentIdx) + "_current_viewport.png")

        # JSON rendering
        uiJSON = ui.renderJSON()
        response["ui"] = uiJSON

        # Store most recent task progress
        self.taskProgress = uiJSON["taskProgress"]
        # Store most recent number of steps
        self.steps = uiJSON["world_steps"]

        # Return response
        return response

    # Returns true if all tasks are marked as complete, and false otherwise.
    def areTasksComplete(self):
        # If no tasks are defined, then return incomplete
        if (len(self.taskProgress) == 0):
            return False

        # Check if all tasks are complete
        for task in self.taskProgress:
            if (task["completed"] == False):
                return False

        # If we reach here, all tasks were marked as complete
        return True

    # Returns a detailed task scorecard (that contains oracle knowledge, and should never be shown to an agent)
    def getTaskScorecard(self):
        # If no agents, then no task progress -- return None
        if (len(self.ui) <= 0):
            return None

        # Get a reference to any agent's UI
        ui = self.ui[0]
        return ui.getFullTaskProgressJSON()

    # Get the current step counter
    def getStepCounter(self):
        return self.steps

    # Lists all known actions (as they are parsed by the JSON action interpreter), by directly enumerating the ActionType enum
    def listKnownActions(self, limited:bool=False):
        return getActionDescriptions(limited)      # from ActionHistory

    # Additional helpful information about the action string, for building prompts.
    def additionalActionDescriptionString(self):
        outStr = ""
        #outStr += "Actions are expressed as JSON. The format is as follows: `{\"action\": \"USE\", \"arg1\": 5, \"arg2\": 12}`, where 'action' is the action type, and 'arg1' and 'arg2' refer to the UUIDs of the objects that serve as arguments. Some actions may require arg1, arg2, or no arguments.  Discovery Feed actions require different arguments.  What arguments are required for specific actions is provided in the known actions list.  Attempting actions not in the known actions list will result in an error."
        outStr += "Actions are expressed as JSON. The format is as follows: `{\"action\": \"USE\", \"arg1\": 5, \"arg2\": 12}`, where 'action' is the action type, and 'arg1' and 'arg2' refer to the UUIDs of the objects that serve as arguments. Some actions may require arg1, arg2, or no arguments.  Some actions, like MOVE_DIRECTION, ROTATE_DIRECTION, and Discovery Feed actions require different arguments, shown above.  What arguments are required for specific actions is provided in the known actions list above.  Attempting actions not in the known actions list, or providing incorrect arguments, will result in an error."
        return outStr

    def listTeleportLocationsDict(self):
        return self.world.getTeleportLocations()

    # Check to see if the agent is currentinly in the middle of a dialog
    def isAgentInDialog(self, agentIdx):
        # Check to make sure the agent index is valid
        if (agentIdx < 0) or (agentIdx >= self.numUserAgents):
            #response = {"errors": ["Agent index out of range. Specified agent index: " + str(agentIdx) + ". Number of agents: " + str(self.numUserAgents) + " (i.e. value must be between 0 and " + str(self.numUserAgents - 1) + ")"]}
            return False

        # Get a reference to this agent's UI
        ui = self.ui[agentIdx]

        # Get a reference to the agent
        agent = ui.currentAgent

        # Return whether or not the agent is in a dialog
        return agent.isInDialog()



    # Perform an action for a given agent
    def performAgentAction(self, agentIdx, actionJSON):
        # Check to make sure the agent index is valid
        if (agentIdx < 0) or (agentIdx >= self.numUserAgents):
            response = {"errors": ["Agent index out of range. Specified agent index: " + str(agentIdx) + ". Number of agents: " + str(self.numUserAgents) + " (i.e. value must be between 0 and " + str(self.numUserAgents - 1) + ")"]}
            return response

        # Check that the world is initialized
        if (self.world == None):
            response = {"errors": ["World is not initialized"]}
            return response

        # Check that this agent has not already acted this step
        if (agentIdx in self.agentsThatHaveActedThisStep):
            response = {"errors": ["Agent has already acted this step.  World tick must be called before this agent can act again."]}
            return response


        # Get a reference to this agent's UI
        ui = self.ui[agentIdx]

        # Get a reference to the agent
        agent = ui.currentAgent

        # Parse any action keys
        doTick, jsonParseErrors, success = ui.parseActionJSON(jsonIn = actionJSON)

        # Pack the response
        response = {
            "errors": jsonParseErrors,
            "success": success
        }

        if (success != None):
            # Update the UI with the message (for the bottom of the screen)
            ui.updateLastActionMessage(success.message)

            # If the message was rated as high importance, also add it to the explicit modal dialog queue
            if (success.importance == MessageImportance.HIGH):
                ui.addTextMessageToQueue(success.message)
        else:
            # Update the UI with the message (for the bottom of the screen)
            ui.updateLastActionMessage("Invalid action")
            response["errors"].append("Invalid action")


        # Mark that this agent has acted this step
        self.agentsThatHaveActedThisStep.add(agentIdx)

        # Hacky: Reformat "success" key so it's JSON serializable.
        # Also removes some errant JSON parse errors (when the action was successful) that are not actually errors.
        # These are caused by the default behavior of the API expecting arguments to be object UUIDs.
        wasSuccessful = response["success"].success
        if (wasSuccessful == True):
            response["errors"] = []
            response["success"] = True
        else:
            response["success"] = response["success"].success


        return response



    #
    #   World tick
    #
    def tick(self):
        response = {
            "errors": [],
            "success": False
        }

        # Check that the world is initialized
        if (self.world == None):
            response["errors"].append("World is not initialized")
            return response

        # Tick the world
        self.world.tick()

        # Reset the agents that have acted this step
        self.agentsThatHaveActedThisStep = set()

        response["success"] = True
        return response



    #
    #   UI
    #
    def setUI(self, agentIdx):
        print("TODO!!!!!!!!!!!!!!!!!")
        pass


    #
    #   Supporting functions
    #



    # Create a headless window -- i.e. a surface that can be used to render to, but no real window is created
    def createHeadlessWindow(self, windowMode="small"):
        # Initialize pygame in headless mode (i.e. no real window is created)
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Game parameters
        gameParams = {
            #"height": 800,
            #"height": 750,
            #"width": 800,
            "height": 1024 + 300,
            "width": 24*32*2,
            "fps": 60,
            "name": "DiscoveryWorld"
        }

        # Create the window
        window = pygame.display.set_mode((gameParams["width"], gameParams["height"]))
        pygame.display.set_caption("DiscoveryWorld")

        # Inititalize the clock
        clock = pygame.time.Clock()

        # Initialize font renderer
        pygame.font.init()

        # Return the window
        return window


    # Convert a directory of frames into an agent video, using ffmpeg
    def createAgentVideo(self, agentIdx:int, filenameOut:str):
        # Call FFMPEG (forces overwrite)
        filenameInPrefix = self.FRAME_DIR + "/ui_agent_" + str(agentIdx) + "_frame_%d.png"
        #filenameOut = "output_agent" + str(agentIdx) + ".mp4"
        # ./ffmpeg -y -framerate 2 -i video/frames-thread1/ui_agent_0_frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p temp.mp4
        # ./ffmpeg -y -framerate 2 -i video/frames-thread7210287/ui_agent_0_frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p temp.mp4
        subprocess.call(["ffmpeg", "-y", "-framerate", "2", "-i", filenameInPrefix, "-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-pix_fmt", "yuv420p", filenameOut])
