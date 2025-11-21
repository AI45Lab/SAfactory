

import os
import random
from discoveryworld.ActionSuccess import ActionSuccess, MessageImportance
from discoveryworld.Agent import NPC, Agent
from discoveryworld.DialogTree import DialogMaker
from discoveryworld.Layer import Layer
from discoveryworld.Pathfinding import Pathfinder
from discoveryworld.buildings.colony import mkGeneralStore, mkKeyShop, mkPaintShop, mkSchool
from discoveryworld.buildings.farm import mkMushroomAndFlowerFarm
from discoveryworld.buildings.terrain import mkFenceX, mkFenceY, mkGrassFill, mkPathX, mkPathY, mkSignVillage, mkTownSquare

from termcolor import colored


ROSETTA_FRENCH = {
    "Bring me the stick!": "Va chercher le bâton!",

    "[Bring me]": "Rapporte-moi",
    "[Add]": "Ajoute",
    "[Reset]": "Remise à zéro",

    "[the stick]": "le bâton",
    "[stick]": "bâton",
    "[key]": "clé",
    "[flower]": "fleur",
    "[mushroom]": "champignon",

    "[red]": "rouge",
    "[green]": "vert",
    "[blue]": "bleu",
    "[yellow]": "jaune",
    "[pink]": "rose",
    "[white]": "blanc",
    "[orange]": "orange",

    "[one]": "un",
    "[two]": "deux",
    "[three]": "trois",
    "[four]": "quatre",
    "[five]": "cinq",

    "[Paint Shop]": "Magasin de peinture",
    "[General Store]": "Magasin général",
    "[Key Shop]": "Magasin de clés",
    "[School]": "École",

    "[seed]": "graine",
    "[shovel]": "pelle",
    "[pot]": "pot",
    "[jar]": "bocal",
    "[flag]": "drapeau",
    "[flowerpot]": "pot de fleurs",
}

ROSETTA_GIBBERISH = {
    "Bring me the stick!": "Zorgle me flibber blonk!",

    "[Bring me]": "Womple",
    "[Add]": "Plonk",
    "[Reset]": "Flibberwomp",

    "[the stick]": "blonk",
    "[stick]": "blonk",
    "[key]": "squibble",
    "[flower]": "florpt",
    "[mushroom]": "nushblort",

    "[red]": "blarg",
    "[green]": "flib",
    "[blue]": "womp",
    "[yellow]": "zibble",
    "[pink]": "quark",
    "[white]": "snoof",
    "[orange]": "gleep",

    "[one]": "blip",
    "[two]": "blob",
    "[three]": "blorp",
    "[four]": "squarp",
    "[five]": "zorp",

    "[Paint Shop]": "Blorpt Vloxurn",
    "[General Store]": "Flibble Vloxurn",
    "[Key Shop]": "Squibble Vloxurn",
    "[School]": "Plonken",

    "[seed]": "snarf",
    "[shovel]": "blurf",
    "[pot]": "plink",
    "[jar]": "klonk",
    "[flag]": "flarp",
    "[flowerpot]": "florptink",
}


class NPCDog(NPC):
    # Constructor
    def __init__(self, world, name, ownerNPC, stick):
        # Default sprite name
        super().__init__(world, name, defaultSpriteName="enemy06_04_agent_facing_west")

        # Rendering
        self.attributes["faceDirection"] = "west"
        self.spriteCharacterPrefix = "enemy06_04_"

        # Default attributes
        self.attributes["isMovable"] = False                       # Dog cannot be picked.
        self.attributes['isContainer'] = True                      # Dog's inventory is a container.
        self.attributes['isOpenable'] = False                      # Can be opened
        self.attributes['isOpenContainer'] = False                 # Prevent other players from accessing its inventory.
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Dialog attributes
        self.attributes['isDialogable'] = False                    # Can it be dialoged with?

        # NPC States
        self.attributes['dialogAgentsSpokenWith'] = []             # List of dialog agents that this NPC has spoken with

        self.pathfinder = Pathfinder()

        self.stick = stick
        self.ownerNPC = ownerNPC
        self.dogIsReady = True  # By default, the dog is ready to fetch stick.
        self.initialLocation = None

    def tick(self):
        # Stop if the object has already had tick() called this update -- this might have happened if the object moved locations in this current update cycle.
        if (self.tickCompleted):
            return

        # Debug
        #print(f"NPCDog (id: {self.name}): {self.attributes['states']}")
        super().tick()

        # Interpret any external states
        if "fetchSignal" in self.attributes['states']:
            self.removeState("wandering")  # Stop wandering.

            # Head towards stick.
            self.attributes["goalLocation"] = (self.stick.attributes["gridX"], self.stick.attributes["gridY"])
            self.removeState("fetchSignal")
            self.addState("movingToStick")

        elif "retrieveStick" in self.attributes['states']:
            self.removeState("wandering")  # Stop wandering.

            # Head towards towards owner.
            self.attributes["goalLocation"] = (self.ownerNPC.attributes["gridX"], self.ownerNPC.attributes["gridY"])
            self.removeState("retrieveStick")
            self.addState("movingToOwner")

        elif "takeStick" in self.attributes['states']:
            objectsAtLocation = self.world.getObjectsAt(self.attributes["gridX"], self.attributes["gridY"])
            # Print names of objects in front of agent
            #print("Objects in front of agent: " + str([x.name for x in objectsAtLocation]))

            # Loop through all objects at that location, looking for the stick
            stickObjects = [x for x in objectsAtLocation if "stick" in x.name]
            # Print names of edible objects in front of agent
            #print("Stick objects in front of agent: " + str([x.name for x in stickObjects]))

            if (len(stickObjects) > 0):
                #print("I want to take the " + stickObjects[0].name)
                # Take the first edible object
                stickObject = stickObjects[0]
                # Take the object
                successTake = self.actionPickUp(stickObject)
                #print(successTake)
                self.stick = stickObject

                self.removeState("takeStick")
                self.addState("retrieveStick")

        elif ("giveStick" in self.attributes['states']):
            success = self.actionPut(self.stick, self.ownerNPC, bypassClosedContainerCheck=True)
            self.removeState("giveStick")
            self.addState("wandering")
            self.dogIsReady = True

        else:
            # Default behavior, if no other behaviors are present, is to wander
            if ("wandering" not in self.attributes['states']):
                self.addState("wandering")

        if "movingToOwner" in self.attributes['states']:
            # Update the goalLocation in case the owner has moved
            self.attributes["goalLocation"] = (self.ownerNPC.attributes["gridX"], self.ownerNPC.attributes["gridY"])

        # Pathfinding/Auto-navigation
        if ("goalLocation" in self.attributes):
            success = self._doNPCAutonavigation()
            # If we're in the "movingToCafeteria" state, check to see if we're already in the goal location
            if ("movingToStick" in self.attributes['states']):
                if (self.attributes["gridX"] == self.attributes["goalLocation"][0]) and (self.attributes["gridY"] == self.attributes["goalLocation"][1]):
                    # The dog is at the stick -- grab it!
                    self.removeState("movingToStick")
                    self.addState("takeStick")
                    # Remove the goal location
                    del self.attributes["goalLocation"]

            elif ("movingToOwner" in self.attributes['states']):
                if (self.attributes["gridX"] == self.attributes["goalLocation"][0]) and (self.attributes["gridY"] == self.attributes["goalLocation"][1]):
                    self.removeState("movingToOwner")
                    self.addState("giveStick")
                    # Remove the goal location
                    del self.attributes["goalLocation"]

        else:
            if ("wandering" in self.attributes['states']):
                #self.attributes["goalLocation"] = (random.randint(0, self.world.sizeX - 1), random.randint(0, self.world.sizeY - 1))
                pass

        #print(f"NPCDog (id: {self.name}): {self.attributes}")


class NPCDogTrainer(NPC):
    def __init__(self, world, name, dogNPC, stick, throwLocation):
        # Default sprite name
        super().__init__(world, name, defaultSpriteName="character35_agent_facing_east")

        # Rendering
        self.attributes["faceDirection"] = "east"
        self.spriteCharacterPrefix = "character35_"

        # Default attributes
        self.attributes["isMovable"] = False                       # Dog cannot be picked.
        self.attributes['isContainer'] = True                      # Dog's inventory is a container.
        self.attributes['isOpenable'] = False                      # Can be opened
        self.attributes['isOpenContainer'] = False                 # Prevent other players from accessing its inventory.
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Dialog attributes
        self.attributes['isDialogable'] = True                    # Can it be dialoged with?

        # NPC States
        self.attributes['dialogAgentsSpokenWith'] = []             # List of dialog agents that this NPC has spoken with

        self.pathfinder = Pathfinder()

        self.stick = stick
        self.dogNPC = dogNPC
        self.trainerIsReady = True  # By default, the trainer is ready to throw the stick.
        self.throwLocation = throwLocation

    def tick(self):
        # Stop if the object has already had tick() called this update -- this might have happened if the object moved locations in this current update cycle.
        if (self.tickCompleted):
            return

        # Debug
        #print(f"NPCDogTrainer (id: {self.name}): {self.attributes['states']}")
        super().tick()

        # Interpret any external states
        hasTheStick = self.stick in self.contents
        #print(colored(f"Trainer has the stick: {hasTheStick}", "cyan"))
        #print(colored(f"Trainer is ready: {self.trainerIsReady}", "cyan"))
        if self.trainerIsReady and hasTheStick:
            self.trainerIsReady = False
            self.addState("throwSignal")

        if "throwSignal" in self.attributes['states']:
            self.removeState("wandering")  # Stop wandering.

            # Throw the stick halfway between the trainer and the dog.
            # throwX = (self.attributes["gridX"] + self.dogNPC.attributes["gridX"]) // 2
            # throwY = (self.attributes["gridY"] + self.dogNPC.attributes["gridY"]) // 2
            self.actionThrow(self.stick, self.throwLocation)

            self.removeState("throwSignal")
            self.addState("tellDogToFetch")

        elif "tellDogToFetch" in self.attributes['states']:
            self.removeState("wandering")  # Stop wandering.

            # Says something like "Go get it, Fido!" TODO: random event with message doesn't work.
            # ActionSuccess(True, f"[Va chercher], {self.dogNPC.name}!", importance=MessageImportance.HIGH)

            self.removeState("tellDogToFetch")
            self.dogNPC.addState("fetchSignal")
            self.addState("waitingForStick")

        elif "waitingForStick" in self.attributes['states']:
            if self.stick not in self.contents:
                self.removeState("waitingForStick")
                self.trainerIsReady = True
                #self.addState("waitingForDogToBark")

        #print(f"NPCDogTrainer (id: {self.name}): {self.attributes}")


class NPCElder(NPC):
    def __init__(self, world, name, scoringInfo):
        # Default sprite name
        super().__init__(world, name, defaultSpriteName="character9_agent_facing_south")

        # Rendering
        self.attributes["faceDirection"] = "south"
        self.spriteCharacterPrefix = "character9_"

        # Default attributes
        self.attributes["isMovable"] = False                       # Elder cannot be picked.
        self.attributes['isContainer'] = True                      # Elder's inventory is a container.
        self.attributes['isOpenable'] = False                      # Can't be opened
        self.attributes['isOpenContainer'] = False                 # Prevent other players from accessing its inventory.
        self.attributes['containerPrefix'] = "on"                  # Container prefix (e.g. "in" or "on")

        # Dialog attributes
        self.attributes['isDialogable'] = True                    # Can it be dialoged with?

        self.pathfinder = Pathfinder()
        self.scoringInfo = scoringInfo
        self.interestingItems = set()

    def enteringDialogWith(self, agentTalkingTo):
        # Check if the agentTalkingTo has the relevant items.
        for obj in agentTalkingTo.getInventory():
            if self.scoringInfo["item"] in obj.name:
                if self.scoringInfo["learningColor"] and self.scoringInfo["color"] not in obj.name.split(" "):
                    continue  # Color doesn't match

                # Take the object from the agentTalkingTo
                self.addState("interestingItems")
                self.interestingItems.add(obj)


    def tick(self):
        if (self.tickCompleted):
            return

        # Debug
        #print(f"NPCElder (id: {self.name}): {self.attributes['states']}")
        super().tick()

        # Interpret any external states
        if "takeItems" in self.attributes['states']:
            # Check if the elder has the relevant items.
            for obj in self.interestingItems:
                # Take the object from the agentTalkingTo
                obj.parentContainer.actionPut(obj, self, bypassClosedContainerCheck=True)

            self.removeState("takeItems")
            self.removeState("interestingItems")
            self.interestingItems.clear()

def translate(text, rosetta):
    for key in rosetta:
        text = text.replace(key, rosetta[key])

    # If there are any brackets left, print warning
    if "[" in text or "]" in text:
        #print(colored(f"Untranslated text: {text}", "red"))
        pass

    return text


def makeScenarioRosettaStone(world, numUserAgents=1, difficulty="easy"):

    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []

    rosetta = ROSETTA_GIBBERISH
    if os.environ.get("FRENCH", False):
        rosetta = ROSETTA_FRENCH

    # Make the Rosetta Stone scenario
    rng = world.rng

    ITEMS = ["mushroom", "flower", "key"]
    COLORS = ["red", "green", "blue", "yellow", "pink", "white", "orange"]
    COUNTS = [1, 2, 3, 4, 5]
    COUNT_WORDS = ["zero", "one", "two", "three", "four", "five"]

    scoringInfo["learningColor"] = False
    scoringInfo["learningCount"] = False
    taskInstruction = None
    if difficulty == "easy":
        ITEMS = ["stick"] + ITEMS + ["shovel"]
    elif difficulty == "medium":
        if world.randomSeed % 2 == 0:
            scoringInfo["learningColor"] = True  # Half the seeds will be about learning colors.
        else:
            scoringInfo["learningCount"] = True  # Half the seeds will be about learning counting.
    elif difficulty == "hard":
        scoringInfo["learningColor"] = True
        scoringInfo["learningCount"] = True

    scoringInfo["color"] = None
    if scoringInfo["learningColor"]:
        scoringInfo["color"] = rng.choice(COLORS)
        scoringInfo["criticalHypotheses"].append(translate(f"The word '[{scoringInfo['color']}]' means '{scoringInfo['color']}'", rosetta))
        scoringInfo["criticalQuestions"].append("Does it clearly state that: " + translate(f"The word '[{scoringInfo['color']}]' means '{scoringInfo['color']}'", rosetta) + "?")

    scoringInfo["count"] = 1
    if scoringInfo["learningCount"]:
        scoringInfo["count"] = rng.choice(COUNTS)
        scoringInfo["countWord"] = COUNT_WORDS[scoringInfo["count"]]
        scoringInfo["criticalHypotheses"].append(translate(f"The word '[{scoringInfo['countWord']}]' means '{scoringInfo['countWord']}'", rosetta))
        scoringInfo["criticalQuestions"].append("Does it clearly state that: " + translate(f"The word '[{scoringInfo['countWord']}]' means '{scoringInfo['countWord']}'", rosetta) + "?")

    scoringInfo["item"] = ITEMS[world.randomSeed % len(ITEMS)]
    scoringInfo["criticalHypotheses"].append(translate(f"The word '[Bring me]' means 'bring me'", rosetta))
    scoringInfo["criticalHypotheses"].append(translate(f"The word '[{scoringInfo['item']}]' means '{scoringInfo['item']}'", rosetta))
    scoringInfo["criticalQuestions"].append("Does it clearly state that: " + translate(f"The word '[Bring me]' means something like 'bring me'", rosetta) + "?")
    scoringInfo["criticalQuestions"].append("Does it clearly state that: " + translate(f"The word '[{scoringInfo['item']}]' means '{scoringInfo['item']}'", rosetta) + "?")

    if scoringInfo["learningCount"] and scoringInfo["learningColor"]:
        taskInstruction = f"[Bring me] [{scoringInfo['countWord']}] [{scoringInfo['color']}] [{scoringInfo['item']}]!"

    elif scoringInfo["learningCount"]:
        taskInstruction = f"[Bring me] [{scoringInfo['countWord']}] [{scoringInfo['item']}]!"

    elif scoringInfo["learningColor"]:
        taskInstruction = f"[Bring me] [{scoringInfo['color']}] [{scoringInfo['item']}]!"

    else:
        taskInstruction = f"[Bring me] [{scoringInfo['item']}]!"

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 1
    if (numUserAgents > MAX_NUM_AGENTS):
        raise ValueError("RosettaStone scenario supports only a single agent at the moment.")

    # Populate with structures/objects
    # Fill with grass
    mkGrassFill(world)

    coloredMushrooms, coloredFlowers = mkMushroomAndFlowerFarm(20, 3, world)

    # Buildings
    mkKeyShop(9, 21, world)
    colorSigns, paintShopBounds = mkPaintShop(19, 21, world)
    mkGeneralStore(7, 4, world)
    countingComputer, resetDisk, measuringTape, flagpole, schoolBounds = mkSchool(19, 7, world)

    scoringInfo["paintShopBounds"] = paintShopBounds
    scoringInfo["colorSign"] = colorSigns['red']

    scoringInfo["schoolBounds"] = schoolBounds
    scoringInfo["flagpole"] = flagpole
    scoringInfo["countingComputer"] = countingComputer
    scoringInfo["resetDisk"] = resetDisk
    scoringInfo["measuringTape"] = measuringTape

    # We get an instruction in an alien language
    # e.g. You need the blue key to unlock the door.
    # e.g. Open/close chest/door
    # e.g. lock/unlock chest/door with color key
    # e.g. give water/object to someone
    # e.g. understand numbers from hearing clock (requires time system)
    # e.g. understand numbers from looking at a rock/mushromom/cats/ collection.
    # e.g. understand numbers from using a dispenser machine: you press the button and get that many items.
    # Open/close is learned from flipping a lever and hearing a click, need to go and see the outcome.
    #

    # Tasks:
    #   put  [X] in/on [Y],
    #   open [X] with [Y],
    #   give [X] to [y]
    #   take [X] of [Y]
    # Store selling keys
    # Store selling color paint

    mkTownSquare(16, 18, world)

    # Paths
    mkPathY(17, 1, 30, world)       # Top/bottom, through town square
    mkPathX(12, 28, 11, world)      # Bottom, along key and paint shops.
    mkPathX(17, 19, 15, world)      # To the east
    mkPathX(1, 19, 16, world)       # To the west
    mkPathY(11, 18, 1, world)       # To general store.
    mkPathY(23, 18, 1, world)       # To school.
    mkPathY(22, 27, 1, world)       # To paint shop.
    mkPathY(12, 27, 1, world)       # To key shop.

    # Fences
    # Top-left corner
    mkFenceY(6, 2, 16, world)
    mkFenceX(6, 2, 10, world)

    # Bottom-left corner
    mkFenceY(6, 21, 8, world)
    mkFenceX(6, 29, 10, world)

    # Bottom-right corner
    mkFenceX(19, 29, 10, world)
    mkFenceY(28, 21, 8, world)

    # Top-right corner
    mkFenceX(19, 2, 10, world)
    mkFenceY(28, 2, 16, world)


    # Add big village sign
    mkSignVillage(16, 2, world)
    mkSignVillage(16, 29, world)

    # Add some plants
    world.addObject(15, 1, Layer.OBJECTS, world.createObject("PlantGeneric"))

    plantCount = 0
    minPlants = 15
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = [obj.type for obj in objs]
        # Check to see that there is grass here
        if ("grass" in objTypes):
            # Check that there is not other things here
            if (len(objTypes) == 1):
                # Add a plant
                world.addObject(randX, randY, Layer.OBJECTS, world.createObject("PlantGeneric"))
                plantCount += 1

    toPlace = []
    for c in COLORS:
        for i in range(4):
            toPlace.append(world.createObject("ColoredMushroom", color=c))
            coloredMushrooms[c].append(toPlace[-1])
            toPlace.append(world.createObject("ColoredFlower", color=c))
            coloredFlowers[c].append(toPlace[-1])

    while toPlace:
        # Pick a random location
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)

        # If outside town delimited by fences, add a plant
        if ((randX < 6 or 28 < randX) or (randY < 2 or 29 < randY)):
            # Check to see if there are any objects other than grass there
            objs = world.getObjectsAt(randX, randY)
            objTypes = [obj.type for obj in objs]
            # Check to see that there is only grass here.
            if objTypes == ["grass"]:
                world.addObject(randX, randY, Layer.OBJECTS, toPlace.pop())

    # DialogMaker
    dialogMaker = DialogMaker()

    # Add some number of user agents
    userAgent = Agent(world)
    # userAgent.addObject(world.createObject("Coin"))
    stick = world.createObject("Stick")
    # userAgent.addObject(stick)
    # Add the agent to a specfic location
    world.addObject(17, 0, Layer.AGENT, userAgent)      # Top Town Entrance
    # world.addObject(16, 18, Layer.AGENT, userAgent)      # Town Square
    # world.addObject(12, 24, Layer.AGENT, userAgent)      # In key shop
    # world.addObject(23, 10, Layer.AGENT, userAgent)      # In school

    # Register the agent with the World so we can keep track of it
    world.addAgent(userAgent)

    # Add teleport locations to world
    world.addTeleportLocation("key shop", 12, 24)
    world.addTeleportLocation("paint shop", 22, 24)
    world.addTeleportLocation("town square", 16, 18)
    world.addTeleportLocation("general store", 11, 10)
    world.addTeleportLocation("school", 23, 10)

    # Add the dog
    dog = NPCDog(world, "Fido", None, stick)
    world.addObject(27, 1, Layer.AGENT, dog)
    world.addAgent(dog)

    dogTrainer = NPCDogTrainer(world, "Trainer", dog, stick, (25, 1))
    dogTrainer.addObject(stick)
    dog.ownerNPC = dogTrainer
    world.addObject(20, 1, Layer.AGENT, dogTrainer)
    world.addAgent(dogTrainer)

    dialogMaker.mkDialogDogTrainer(dogTrainer, message=translate("[Bring me] [the stick]!", rosetta))

    elder = NPCElder(world, "Elder", scoringInfo)
    world.addObject(18, 18, Layer.AGENT, elder)
    world.addAgent(elder)
    dialogMaker.mkDialogElder(elder, message=translate(taskInstruction, rosetta))
    scoringInfo["criticalHypotheses"].append(f"The elder says '{translate(taskInstruction, rosetta)}' which translates to '{taskInstruction.replace('[', '').replace(']', '')}")

    scoringInfo["elder"] = elder

    # Apply the Rosetta Stone translation to all signs in the world.
    for obj in world.getAllWorldObjects():
        if obj.attributes.get("document") is not None:
            obj.attributes["document"] = translate(obj.attributes["document"], rosetta)

        if obj.type == "floppy disk":
            obj.name = translate(obj.name, rosetta)

    return scoringInfo