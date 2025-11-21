# Pathfinding.py

import random
import math

from enum import Enum

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from discoveryworld.Layer import Layer
from discoveryworld.ActionSuccess import ActionResult


class Pathfinder():
    # Constructor
    def __init__(self):
        pass

    # Find a path
    # Returns the next step to take on the path, as well as the total length of the path.
    def findPathNextStep(self, world, startX, startY, endX, endY):

        # Step 1: Construct a cost grid representing the world, where each cell contains whether it's passable or not
        costGrid = []
        for y in range(world.sizeY):
            costGrid.append([])
            for x in range(world.sizeX):

                # By default, moving to the next cell has a base cost of 1
                passableScore = 1

                # Check if the cell is passable
                allObjs = world.getObjectsAt(x, y)

                # Check if any of the objects are impassable
                for object in allObjs:
                    if (not object.attributes["isPassable"]):
                        # Check to see if the object is a door, and therefore potentially passable
                        if (object.attributes["isPassage"]):
                            # Potentially passable, with a cost
                            passableScore = 20
                        else:
                            # Not passable (score of -1)
                            passableScore = -1

                        break
                # If this tile has any agents in it, increase its traversal cost
                if (passableScore > 0):
                    if (len(world.grid[x][y]["layers"][Layer.AGENT]) > 0):
                        passableScore += 10


                # Store the passability score
                costGrid[y].append(passableScore)

        # Step 2: Find a path using A*
        print("Finding path from (" + str(startX) + ", " + str(startY) + ") to (" + str(endX) + ", " + str(endY) + ")")

        # Use A* library to find a path
        grid = Grid(matrix=costGrid)

        start = grid.node(startX, startY)
        end = grid.node(endX, endY)

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        #print('operations:', runs, 'path length:', len(path))
        #print(grid.grid_str(path=path, start=start, end=end))

        # Get the next step in the path
        if (len(path) > 1):
            nextX = path[1].x
            nextY = path[1].y
            return (True, nextX, nextY, len(path))
        else:
            return (False, -1, -1, -1)


    #
    #   Autopilot Functions
    #

    # Run the next step of a given autopilot action.
    # Arguments:
    #   autopilotAction: The action to run
    #   agent: The agent running the action
    #   world: The world the agent is in
    # Returns:
    #   success: True (action running successfully) or False (action completed)
    def actionInterpreter(self, autopilotAction, agent, world):
        # First, figure out which type of action we're running
        actionType = autopilotAction.actionType
        # Then, run the appropriate function
        if (actionType == AutopilotActionType.GOTO_XY):
            result = self.runGotoXY(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.PICKUP_OBJ):
            result = self.runPickupObj(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.PLACE_OBJ_IN_CONTAINER):
            result = self.runPlaceObjInContainer(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.DROP_OBJ_AT_LOCATION):
            result = self.runDropObjAtLocation(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.FIND_OBJS_AREA_PLACE):
            result = self.runFindObjsInAreaThenPlace(autopilotAction, autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.LOCATE_BLANK_TILE_IN_AREA):
            result = self.runBlankTileInAreaThenMove(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.WANDER):
            result = self.runWander(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.WANDER_SLOW):
            result = self.runWander(autopilotAction.args, agent, world, probabilityToMove=0.25)
        elif (actionType == AutopilotActionType.WAIT):
            result = self.runWait(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.EAT_OBJ_IN_INVENTORY):
            result = self.runEat(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.DIG_IN_FRONT_OF_AGENT):
            result = self.runDigInFrontOfAgent(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.BURY_IN_FRONT_OF_AGENT):
            result = self.runBuryInFrontOfAgent(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.POST_DISCOVERY_FEED_UPDATE):
            result = self.runPostDiscoveryFeedUpdate(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.MOVE_RELATIVE):
            result = self.runMoveRelative(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.ROTATE_TO_FACE_DIRECTION):
            result = self.runRotateToFaceDirection(autopilotAction.args, agent, world)
        elif (actionType == AutopilotActionType.CHECK_CONDITION):
            result = self.runCheckCondition(autopilotAction.args, agent, world)
        else:
            print("ERROR: Invalid autopilot action type: " + str(actionType))
            return ActionResult.INVALID

        # Check if there's a callback in the args
        if ("callback" in autopilotAction.args) and (autopilotAction.args["callback"] != None):
            # Run the callback
            print("Running callback")
            # Append 'result' to existing parameters in callback
            callbackArgs = []
            if ("callbackArgs" in autopilotAction.args):
                # Copy existing params
                callbackArgs = autopilotAction.args["callbackArgs"]
            # Append result
            callbackArgs.append(result)
            # Run callback
            autopilotAction.args["callback"](callbackArgs)

        return result

    # Check a condition through a callback to the agent
    def runCheckCondition(self, args:dict, agent, world):
        #self.args['conditionCallback'] = conditionCallback
        #self.args['addActionIfTrue'] = addActionIfTrue
        #self.args['addActionIfFalse'] = addActionIfFalse

        # First, call the callback (which takes no arguments).  It should return a boolean
        conditionCallback = args['conditionCallback']
        conditionResult = conditionCallback()

        # If the condition is true, add the 'addActionIfTrue' action to the queue
        if (conditionResult):
            actionIfTrue = args['addActionIfTrue']
            agent.addAutopilotActionToQueue(actionIfTrue)
            return ActionResult.COMPLETED
        else:
            # If the condition is false, add the 'addActionIfFalse' action to the queue
            actionIfFalse = args['addActionIfFalse']
            agent.addAutopilotActionToQueue(actionIfFalse)
            return ActionResult.COMPLETED


    # Do a relative move (north, east, south, west)
    def runMoveRelative(self, args:dict, agent, world):
        direction = args['direction'].lower()

        # Direction should be one of "north", "east", "south", "west"
        if (direction not in ["north", "east", "south", "west"]):
            print("ERROR: Invalid direction specified: " + str(direction))
            return ActionResult.FAILURE

        result = agent.actionMoveAgentNorthEastSouthWest(direction=direction)

        return ActionResult.COMPLETED

    # Rotate to face a specific direction
    def runRotateToFaceDirection(self, args:dict, agent, world):
        direction = args['direction'].lower()

        # Direction should be one of "north", "east", "south", "west"
        if (direction not in ["north", "east", "south", "west"]):
            print("ERROR: Invalid direction specified: " + str(direction))
            return ActionResult.FAILURE

        success = agent.actionRotateAgentFacingDirectionAbsolute(direction)
        return ActionResult.COMPLETED


    def runGotoXY(self, args:dict, agent, world):
        # Check whether we should go to the location, or just go beside it then face it
        goBesideAndFace = args['goBesideAndFace']       # Only one of 'goBesideAndFace' or 'finalDirection' should be specified
        finalDirection = args['finalDirection']

        if (goBesideAndFace == False):
            # First, check if we're already at the destination
            agentLocation = agent.getWorldLocation()
            if (agentLocation[0] == args['destinationX']) and (agentLocation[1] == args['destinationY']):
                # We're already there

                # Check to see if we need to face a specific direction
                if (finalDirection == None):
                    # No direction specified, so just return
                    return ActionResult.COMPLETED

                else:
                    validDirections = ["north", "east", "south", "west"]
                    if (finalDirection not in validDirections):
                        print("ERROR: Invalid direction specified: " + str(finalDirection))
                        return ActionResult.FAILURE

                    # Check to see if we're already facing the correct direction
                    if (agent.attributes["faceDirection"] == finalDirection):
                        # We're already facing the correct direction
                        return ActionResult.COMPLETED
                    else:
                        # We're not facing the final direction, so start rotating
                        success = agent.rotateToFaceDirection(finalDirection)
                        return ActionResult.SUCCESS


            # Otherwise, find the next step in the path
            result = self._doNPCAutonavigation(agent, world, args['destinationX'], args['destinationY'], besideIsOK=True)
            print("runGotoXY: _doNPCAutonavigation returned: " + str(result))

        else:
            # Check if we are directly left, right, up, or down from the object
            agentLocation = agent.getWorldLocation()
            isBeside, besideDir = self._isAgentDirectlyAdjacentToDestination(agentLocation[0], agentLocation[1], args['destinationX'], args['destinationY'])

            if (isBeside):
                print("runGotoXY: We are beside the object (it is to our " + str(besideDir) + ")")

                # We're directly beside the object.  Is the agent facing the object?
                if (agent.attributes["faceDirection"] != besideDir) and (besideDir != "identical"):
                    # We're not facing the object.  Rotate.
                    print("runPickupObj: We're not facing the object.  Rotating to face it.")
                    success = agent.rotateToFaceDirection(besideDir)
                    return ActionResult.SUCCESS

                # We're facing the object.  We're done.
                return ActionResult.COMPLETED
            else:
                result = self._doNPCAutonavigation(agent, world, args['destinationX'], args['destinationY'], besideIsOK=True)
                print("runGotoXY: _doNPCAutonavigation returned: " + str(result))


        # If we reach here, it's running
        return ActionResult.SUCCESS


    # Eat an object in the agent's inventory
    def runEat(self, args:dict, agent, world):
        # First, check that the object to eat is in the agent's inventory
        objectNamesOrTypesToEat = args['objectNamesOrTypesToEat']

        # Check if an object with an appropriate name/type is in the agent's inventory
        agentInventory = agent.getAllContainedObjectsRecursive(respectContainerStatus=True)
        objectToEat = None
        for invObj in agentInventory:
            # Check if this object matches the name or type
            if (invObj.name in objectNamesOrTypesToEat) or (invObj.type in objectNamesOrTypesToEat):
                # Found the object
                objectToEat = invObj
                break

        if (objectToEat == None):
            # Object meeting name/type requirements not found in inventory.
            print("runEat: Object with that name or type (" + str(objectNamesOrTypesToEat) + " not found in agent's inventory.")
            return ActionResult.FAILURE

        # Perform the action
        result = agent.actionEat(objectToEat)

        # Check the result
        if (result.success == False):
            # Action failed
            print("runEat: Action failed (" + result.message + ")")
            return ActionResult.FAILURE

        # Action succeeded
        return ActionResult.COMPLETED


    # Post an update to the Discovery Feed
    def runPostDiscoveryFeedUpdate(self, args:dict, agent, world):
        # Content and signals
        contentStr = args['contentStr']
        signals = args['signals']

        # Post the update to the feed
        agent.actionDiscoveryFeedMakeUpdatePost(contentStr, signals=signals)

        # Action succeeded
        return ActionResult.COMPLETED


    # Bury an object in front of the agent
    def runBuryInFrontOfAgent(self, args:dict, agent, world):
        objectNamesOrTypesToDig = args['objectNamesOrTypesToDig']
        objectNamesOrTypesToBury = args['objectNamesOrTypesToBury']

        # Check if an object with an appropriate name/type is in the agent's inventory
        agentInventory = agent.getInventory()
        objToBury = None
        for invObj in agentInventory:
            if (invObj.name in objectNamesOrTypesToBury) or (invObj.type in objectNamesOrTypesToBury):
                # Found the object
                objToBury = invObj
                break

        if (objToBury == None):
            # Object meeting name/type requirements not found in inventory.
            print("runBuryInFrontOfAgent: Object with that name or type (" + str(objectNamesOrTypesToBury) + " not found in agent's inventory.")
            return ActionResult.FAILURE

        # First, try to dig in front of the agent
        actionDig = AutopilotAction_DigInFrontOfAgent(objectNamesOrTypesToDig, priority=args['priority']+1)
        agent.addAutopilotActionToQueue( actionDig )
        print("Added actionDig to queue: " + str(actionDig))
        ### TODO: WHY DOES THIS NEED TO HAPPEN TWICE?  WHY DOES THE ACTION DISAPPEAR THE FIRST TIME?
        ### ALSO: DIGS ALL 5 TIMES.  POTENTIALLY SOMETHING TO DO WITH THAT??? (e.g. an action unintentionally repeating for all seeds in inventory)


        # Then, try to put the object in the hole
        #class AutopilotAction_PlaceObjInContainer(AutopilotAction):
        #   def __init__(self, objectToPlace, container, containerNamesOrTypes = None, callback=None, callbackArgs=None, priority=3):
        actionPlaceObj = AutopilotAction_PlaceObjInContainer(objToBury, container=None, containerNamesOrTypes=objectNamesOrTypesToDig, priority=args['priority']+1)
        agent.addAutopilotActionToQueue( actionPlaceObj )
        print ("Added actionPlaceObj to queue: " + str(actionPlaceObj))

        # Then, put the dirt back in the hole
        actionPlaceDirt = AutopilotAction_PlaceObjInContainer(objectToPlace=None, container=None, objectNamesOrTypes=["dirt"], containerNamesOrTypes=objectNamesOrTypesToDig, priority=args['priority']+1)
        agent.addAutopilotActionToQueue( actionPlaceDirt )
        print ("Added actionPlaceDirt to queue: " + str(actionPlaceDirt))

        # Complete this action, so we can start the above sequence
        print("runBuryInFrontOfAgent: Returning COMPLETED.")
        return ActionResult.COMPLETED



    # Eat an object in the agent's inventory
    def runDigInFrontOfAgent(self, args:dict, agent, world):
        objectNamesOrTypesToDig = args['objectNamesOrTypesToDig']

        print("### runDigInFrontOfAgent: Digging in front of agent.")

        # Check if an object with an appropriate name/type is in the agent's inventory
        agentInventory = agent.getInventory()
        # Check that the agent has a shovel
        shovel = None
        for invObj in agentInventory:
            if (invObj.type == "shovel"):
                shovel = invObj
                break

        # If no shovel, then break
        if (shovel == None):
            print("### runDigInFrontOfAgent: Agent does not have a shovel.")
            return ActionResult.FAILURE

        # Check if there's a diggable object in front of the agent
        objsAgentFacing = agent.getObjectsAgentFacing()
        # Filter to include only diggable objects
        diggableObjsAgentFacing = []
        for obj in objsAgentFacing:
            if (obj.attributes['isShovelable']):
                diggableObjsAgentFacing.append(obj)

        # Find the first object meeting the criteria
        objToDig = None
        for obj in diggableObjsAgentFacing:
            if (obj.name in objectNamesOrTypesToDig) or (obj.type in objectNamesOrTypesToDig):
                # Found the object
                objToDig = obj
                break

        # Check if we found an object
        if (objToDig == None):
            # Object meeting name/type requirements not found in inventory.
            print("runDigInFrontOfAgent: Object with that name or type (" + str(objectNamesOrTypesToDig) + " not found in front of agent.")
            return ActionResult.FAILURE

        # Perform the action (use shovel on object)
        print("### runDigInFrontOfAgent: Digging object.")
        result = agent.actionUse(objToUse=shovel, objToUseOn=objToDig)
        print("### runDigInFrontOfAgent: Digging object result: " + str(result))

        # Action succeeded
        return ActionResult.COMPLETED


    def runPickupObj(self, args:dict, agent, world):
        # First, get the object's world location
        objectToPickUp = args['objectToPickUp']
        objLocation = objectToPickUp.getWorldLocation()

        # Check if we are directly left, right, up, or down from the object
        agentLocation = agent.getWorldLocation()
        isBeside, besideDir = self._isAgentDirectlyAdjacentToDestination(agentLocation[0], agentLocation[1], objLocation[0], objLocation[1])

        if (isBeside):
            print("runPickupObj: We are beside the object (it is to our " + str(besideDir) + ")")

            # We're directly beside the object.  Is the agent facing the object?
            if (agent.attributes["faceDirection"] != besideDir) and (besideDir != "identical"):
                # We're not facing the object.  Rotate.
                print("runPickupObj: We're not facing the object.  Rotating to face it.")
                success = agent.rotateToFaceDirection(besideDir)
                return ActionResult.SUCCESS
            else:
                # We're facing the object.  Check to see if we can pick it up (i.e. if it's accessible, or in a closed container)
                print("runPickupObj: We're facing the object.  Checking if we can pick it up.")
                objectContainerIfClosed = objectToPickUp.getOutermostClosedContainer()
                if (objectContainerIfClosed != None):
                    # The object is in a closed container.  Open the container.
                    print("runPickupObj: The object is in a closed container (" + objectContainerIfClosed.name + ").  Opening the container.")
                    #success = agent.actionOpenContainer(objectContainerIfClosed)
                    success = agent.actionOpenClose(objectContainerIfClosed, whichAction="open")
                    print("runPickupObj: actionOpenClose returned: " + str(success))
                    return ActionResult.SUCCESS

                # If we reach here, the object is accessible and we can pick it up.  Pick it up.
                success = agent.actionPickUp(objectToPickUp)

                # TODO: Check for success
                return ActionResult.COMPLETED

        else:
            # We're not close enough to pick up the object. Navigate to it.
            result = self._doNPCAutonavigation(agent, world, objLocation[0], objLocation[1], besideIsOK=True)
            print("runPickupObj: _doNPCAutonavigation returned: " + str(result))

            return result


    # Helper: Find the first object meeting a given type within a specified area
    def _findFirstObjectOfType(self, args:dict, agent, world):
        startX = args['x']
        startY = args['y']
        endX = args['x'] + args['width']
        endY = args['y'] + args['height']
        objectTypes = args['objectTypes']

        excludeObjectsOnAgent = args['excludeObjectsOnAgent']

        agentInventory = []
        if (excludeObjectsOnAgent):
            agentInventory = agent.getAllContainedObjectsRecursive(respectContainerStatus=False)

        for tileX in range(startX, endX):
            for tileY in range(startY, endY):
                objectsAtTile = world.getObjectsAt(tileX, tileY, respectContainerStatus=True, excludeObjectsOnAgents=excludeObjectsOnAgent)
                for obj in objectsAtTile:
                    if (obj.type in objectTypes):
                        # We found an object that matches one of the types we're looking for.  Place it in the container.
                        # But, first check that the object isn't in the agents inventory (if we're supposed to exclude objects on the agent)
                        if (excludeObjectsOnAgent):
                            if (obj not in agentInventory):
                                return obj
                        else:
                            return obj

        # If we reach here, we didn't find any objects of the specified type
        return None

    # Helper: Find a list of tiles meeting a given type, that doesn't have any objects on top of it.
    def _findBlankTilesOfType(self, args:dict, agent, world):
        startX = args['x']
        startY = args['y']
        endX = args['x'] + args['width']
        endY = args['y'] + args['height']
        objectTypes = args['objectTypes']
        allowedContentTypes = args['allowedContentTypes']

        # doesTileHaveObjectsOnIt(self, x, y):
        agentInventory = []
        excludeObjectsOnAgent = True
        if (excludeObjectsOnAgent):
            agentInventory = agent.getAllContainedObjectsRecursive(respectContainerStatus=False)

        matches = []
        for tileX in range(startX, endX):
            for tileY in range(startY, endY):
                # First, check if there are any objects on this tile (i.e. not blank)
                objsOnTile = world.doesTileHaveObjectsOnIt(tileX, tileY)
                if (objsOnTile):
                    # There are objects on this tile.  Skip it.
                    continue

                # Then, check if the tile is of the specified type
                objectsAtTile = world.getObjectsAt(tileX, tileY, respectContainerStatus=True)
                for obj in objectsAtTile:
                    if (obj.type in objectTypes):
                        # We found an object that matches one of the types we're looking for.

                        # Check that the object doesn't contain anything except whatever the allowed contents are
                        containsOnlyAllowedContents = True
                        if (allowedContentTypes != None):
                            for cObj in obj.contents:
                                if (cObj.type not in allowedContentTypes):
                                    containsOnlyAllowedContents = False
                                    break

                        if (containsOnlyAllowedContents):
                            # But, first check that the object isn't in the agents inventory (if we're supposed to exclude objects on the agent)
                            if (excludeObjectsOnAgent):
                                if (obj not in agentInventory):
                                    matches.append(obj)
                            else:
                                matches.append(obj)

        # Return any matches we found
        return matches


    def runFindObjsInAreaThenPlace(self, originalAction, args:dict, agent, world):
        # Step 1: Find if there's an object in the area that matches the object type(s) we're looking for (if not, we're done)
        nextObj = self._findFirstObjectOfType(args, agent, world)

        print("##### runFindObjsInAreaThenPlace: Found object: " + str(nextObj))
        # If there are no objects to find, we're done
        if (nextObj == None):
            return ActionResult.COMPLETED

        #self.args['maxToTake'] = maxToTake                              # The maximum number of objects meeting the criteria to take
        #self.args['numTaken'] = 0
        # If the 'maxToTake' parameter is set (i.e. > 0), and we've already taken that many objects, we're done
        if ('maxToTake' in args):
            print("###### runFindObjsInAreaThenPlace: Already taken " + str(args['numTaken']) + " objects, and maxToTake is " + str(args['maxToTake']) + ".  Checking if we're done.")
            if (args['maxToTake'] > 0):
                if ('numTaken' in args):
                    if (args['numTaken'] >= args['maxToTake']):
                        print("###### runFindObjsInAreaThenPlace: Already taken " + str(args['numTaken']) + " objects, and maxToTake is " + str(args['maxToTake']) + ".  Action completed.")
                        return ActionResult.COMPLETED

        # Step 2: If we reach here, we found an object -- place it in the container
        container = args['container']
        priority = args['priority']

        # Pop on a 'placeObjInContainer' action, with a higher priority
        thingToPickup = nextObj
        whereToPlace = container
        actionPick = AutopilotAction_PickupObj(thingToPickup, priority=priority+1)
        #actionPlace = AutopilotAction_PlaceObjInContainer(thingToPickup, whereToPlace, priority=priority+1)
        # Provide a callback to PickupObj that increments the 'numTaken' parameter
        def callbackPickupObj(callbackArgs):
            action = callbackArgs[0]
            result = callbackArgs[1]
            if (result == ActionResult.COMPLETED):
                if ('numTaken' in action.args):
                    action.args['numTaken'] += 1
                else:
                    action.args['numTaken'] = 1

                print("##### runFindObjsInAreaThenPlace: CALLBACK Incremented numTaken to " + str(action.args['numTaken']))

        #actionPlace = AutopilotAction_PlaceObjInContainer(thingToPickup, whereToPlace, priority=priority+1, callback=callbackPickupObj(self))
        # Above doesn't work because the callback is called with an additional 'result' parameter, which is the result of the action that just completed.
        # e.g. autopilotAction.args["callback"](result)
        # Instead, we'll change the call to this:
        actionPlace = AutopilotAction_PlaceObjInContainer(thingToPickup, whereToPlace, priority=priority+1, callback=callbackPickupObj, callbackArgs=[originalAction])


        agent.addAutopilotActionToQueue( actionPick )
        agent.addAutopilotActionToQueue( actionPlace )
        print("##### Added action to queue " + str(actionPick) )
        print("##### Added action to queue " + str(actionPlace) )
        # Suspend the current action, so the 'placeObjInContainer' action can run
        return ActionResult.SUSPEND


    # Find a tile (of one of the types listed in 'objectTypes') that doesn't have any objects on top of it, and go beside it.
    def runBlankTileInAreaThenMove(self, args:dict, agent, world):
        # Step 1: Find if there's an object in the area that matches the object type(s) we're looking for (if not, we're done)
        blankTiles = self._findBlankTilesOfType(args, agent, world)

        print("##### runBlankTileInAreaThenMove: Found " + str(len(blankTiles)) + " blank tiles of type(s): " + str(args['objectTypes']))
        # If there are no objects to find, we're done
        if (len(blankTiles) == 0):
            print("##### runBlankTileInAreaThenMove: No blank tiles found. exiting.")
            # TODO: Should this be an error?
            return ActionResult.COMPLETED

        # If we're beside and facing one of the tiles, then we're done
        objsAgentFacing = agent.getObjectsAgentFacing(respectContainerStatus=True)
        for blankTile in blankTiles:
            if (blankTile in objsAgentFacing):
                print("##### runBlankTileInAreaThenMove: We're already facing a blank tile with the required properties.  Exiting.")
                return ActionResult.COMPLETED

        # If not, then randomly pick one tile
        tileObj = random.choice(blankTiles)
        tileLocation = tileObj.getWorldLocation()

        # Move to that spot
        actionGoto = AutopilotAction_GotoXY(tileLocation[0], tileLocation[1], goBesideAndFace=True, priority=args['priority']+1)
        agent.addAutopilotActionToQueue( actionGoto )
        print("##### Added action to queue " + str(actionGoto) )

        # Suspend the current action, so the 'AutopilotAction_GotoXY' action can run
        return ActionResult.SUSPEND




    def runPlaceObjInContainer(self, args:dict, agent, world):
        # self.args['objectToPlace'] = objectToPlace
        # self.args['container'] = container

        # Object to place
        objectToPlace = args['objectToPlace']
        objectNamesOrTypes = args['objectNamesOrTypes']
        # If objectToPlace is None and objectNames is a list, then we need to find the object in the agents inventory, or directly in front of the agent
        if (objectToPlace == None and objectNamesOrTypes != None):
            # First, check inventory
            objsAgentInventory = agent.getInventory()
            for objAgentInventory in objsAgentInventory:
                if (objAgentInventory.name in objectNamesOrTypes or objAgentInventory.type in objectNamesOrTypes):
                    objectToPlace = objAgentInventory
                    break
            # If we still haven't found it, then we're done
            if (objectToPlace == None):
                print("runPlaceObjInContainer: ERROR: Couldn't find object (" + str(objectNamesOrTypes) + " to place in agent's inventory.  Exiting.")
                return ActionResult.FAILURE

        # Container
        container = args['container']
        containerNamesOrTypes = args['containerNamesOrTypes']
        # If container is None and containerNames is a list, then we need to find the container (directly in front of the agent)
        if (container == None and containerNamesOrTypes != None):
            objsAgentFacing = agent.getObjectsAgentFacing(respectContainerStatus=True)
            for objAgentFacing in objsAgentFacing:
                if (objAgentFacing.name in containerNamesOrTypes or objAgentFacing.type in containerNamesOrTypes):
                    container = objAgentFacing
                    break
            # If we still haven't found it, then we're done
            if (container == None):
                print("runPlaceObjInContainer: ERROR: Couldn't find container (" + str(containerNamesOrTypes) + " in front of agent.  Exiting.")
                return ActionResult.FAILURE

        # Container location
        containerLocation = container.getWorldLocation()

        # Check if we have the object in our inventory.  If not, something unexpected has happened, so return an error.
        agentInventory = agent.getAllContainedObjectsRecursive(respectContainerStatus=True)
        # TODO: Handle case where the object is in a closed container, and we need to open the container to get it
        if (objectToPlace not in agentInventory):
            print("runPlaceObjInContainer: ERROR: I can't seem to find the object I need to place (" + objectToPlace.name + ") in the agent inventory. Stopping place action.")
            return ActionResult.FAILURE
            #return ActionResult.ERROR


        # Check if we are directly left, right, up, or down from the container
        agentLocation = agent.getWorldLocation()
        isBeside, besideDir = self._isAgentDirectlyAdjacentToDestination(agentLocation[0], agentLocation[1], containerLocation[0], containerLocation[1])
        print ("## runPlaceObjInContainer: Agent is at (" + str(agentLocation[0]) + ", " + str(agentLocation[1]) + ")")
        print ("## runPlaceObjInContainer: Container is at (" + str(containerLocation[0]) + ", " + str(containerLocation[1]) + ")")

        if (isBeside):
            print("runPlaceObjInContainer: We are beside the container (it is to our " + str(besideDir) + ")")

            # We're directly beside the object.  Is the agent facing the object?
            if (agent.attributes["faceDirection"] != besideDir) and (besideDir != "identical"):
                # We're not facing the object.  Rotate.
                print("runPlaceObjInContainer: We're not facing the container.  Rotating to face it.")
                success = agent.rotateToFaceDirection(besideDir)
                return ActionResult.SUCCESS
            else:
                # We're facing the container.  If it needs to be opened, open it.
                print("runPlaceObjInContainer: We're facing the container.  Checking if we need to open it.")
                objectContainerIfClosed = container.getOutermostClosedContainer(includeSelf=True)
                if (objectContainerIfClosed != None):
                    # The object is in a closed container.  Open the container.
                    print("runPlaceObjInContainer: The following container is closed and needs to be opened (" + objectContainerIfClosed.name + ").  Opening the container.")
                    #success = agent.actionOpenContainer(objectContainerIfClosed)
                    success = agent.actionOpenClose(objectContainerIfClosed, whichAction="open")
                    print("runPlaceObjInContainer: actionOpenContainer returned: " + str(success))
                    return ActionResult.SUCCESS

                # If we reach here, the object is accessible and we can put it in/on the container.  Move it to the container.
                print("runPlaceObjInContainer: Putting object (" + objectToPlace.name + ") in container (" + container.name + ")")
                success = agent.actionPut(objToPut=objectToPlace, newContainer=container)
                print("runPlaceObjInContainer: actionPut returned: " + str(success))
                print("runPlaceObjInContainer: " + str(container.attributes))

                # TODO: Check for success
                return ActionResult.COMPLETED

        else:
            # We're not close enough to pick up the object. Navigate to it.
            result = self._doNPCAutonavigation(agent, world, containerLocation[0], containerLocation[1], besideIsOK=True)
            print("runPlaceObjInContainer: _doNPCAutonavigation returned: " + str(result))

            return result


    def runDropObjAtLocation(self, args:dict, agent, world):
        # Object to place
        objectToPlace = args['objectToPlace']
        objectNamesOrTypes = args['objectNamesOrTypes']
        # If objectToPlace is None and objectNames is a list, then we need to find the object in the agents inventory, or directly in front of the agent
        if (objectToPlace == None and objectNamesOrTypes != None):
            # First, check inventory
            objsAgentInventory = agent.getInventory()
            for objAgentInventory in objsAgentInventory:
                if (objAgentInventory.name in objectNamesOrTypes or objAgentInventory.type in objectNamesOrTypes):
                    objectToPlace = objAgentInventory
                    break
            # If we still haven't found it, then we're done
            if (objectToPlace == None):
                print("runDropObjAtLocation: ERROR: Couldn't find object (" + str(objectNamesOrTypes) + " to place in agent's inventory.  Exiting.")
                return ActionResult.FAILURE

        # Drop location
        dropLocation = args['dropLocation']

        # Check if we have the object in our inventory.  If not, something unexpected has happened, so return an error.
        agentInventory = agent.getAllContainedObjectsRecursive(respectContainerStatus=True)
        # TODO: Handle case where the object is in a closed container, and we need to open the container to get it
        if (objectToPlace not in agentInventory):
            print("runDropObjAtLocation: ERROR: I can't seem to find the object I need to place (" + objectToPlace.name + ") in the agent inventory. Stopping place action.")
            return ActionResult.FAILURE

        # Check if we are at the location
        agentLocation = agent.getWorldLocation()
        if (agentLocation[0] == dropLocation[0] and agentLocation[1] == dropLocation[1]):
            # We're at the location.  Drop the object.
            print("runDropObjAtLocation: We are at the drop location.  Dropping object.")
            success = agent.actionDrop(objToDrop=objectToPlace)
            print("runDropObjAtLocation: actionDrop returned: " + str(success))
            return ActionResult.COMPLETED

        else:
            # We're not at the location to drop the object. Navigate to it.
            result = self._doNPCAutonavigation(agent, world, dropLocation[0], dropLocation[1], besideIsOK=False)
            print("runDropObjAtLocation: _doNPCAutonavigation returned: " + str(result))

            return result


    def runWander(self, args:dict, agent, world, probabilityToMove=0.5):
        # NEW WANDER
        # Check if the agent is at args['preferredX'], args['preferredY']
        agentLocation = agent.getWorldLocation()
        # Get the euclidian distance to the preferred location
        distToPreferred = math.sqrt( (agentLocation[0] - args['preferredX'])**2 + (agentLocation[1] - args['preferredY'])**2 )

        if (distToPreferred < 4):
            # We're close to the preferred location, so do a low-computation move
            # Only move 50% of the time -- keeps the agent's speed slow when it's wandering, so a user can catch up.
            if (random.random() < probabilityToMove):
                # Randomly pick a direction to move in
                directions = ["north", "east", "south", "west"]
                direction = random.choice(directions)
                action = AutopilotAction_MoveRelative(direction, priority=args['priority']+1)
                agent.addAutopilotActionToQueue(action)

        else:
            # We're getting far from the preferred location -- move back to it.
            # Generate a GOTO action to get there
            print("runWander:  Generating preferred x/y location to wander to (" + str(args['preferredX']) + ", " + str(args['preferredY']) + ")" )
            action = AutopilotAction_GotoXY(args['preferredX'], args['preferredY'], priority=args['priority']+1)
            # Add the action to the agent's queue
            agent.addAutopilotActionToQueue(action)

        return ActionResult.SUCCESS


    def runWait(self, args:dict, agent, world):
        pass



    #
    #   Helper functions
    #

    #
    #   NPC Auto-navigation
    #
    #  This function is used by the NPC to automatically navigate to a given location.
    #  Arguments:
    #   - agent: The agent that is navigating
    #   - world: The world object
    #   - destinationX: The X coordinate of the destination
    #   - destinationY: The Y coordinate of the destination
    #   - besideIsOK: If the agent can't reach the destination, is it OK to reach a location directly N/E/S/W from it?
    def _doNPCAutonavigation(self, agent, world, destinationX, destinationY, besideIsOK=False):

        pathSuccess, nextX, nextY, pathLength = self.findPathNextStep(world, agent.attributes["gridX"], agent.attributes["gridY"], destinationX, destinationY)
        # Back-off strategy: If 'besideIsOK=True', try to find a path to a location directly N/E/S/W from the destination
        if (not pathSuccess) and besideIsOK:
            pathLength = 999999
            # North
            _pathSuccess, _nextX, _nextY, _pathLength = self.findPathNextStep(world, agent.attributes["gridX"], agent.attributes["gridY"], destinationX, destinationY-1)
            if (_pathSuccess) and (_pathLength < pathLength):
                pathSuccess = _pathSuccess
                nextX = _nextX
                nextY = _nextY
                pathLength = _pathLength

            # East
            _pathSuccess, _nextX, _nextY, _pathLength = self.findPathNextStep(world, agent.attributes["gridX"], agent.attributes["gridY"], destinationX+1, destinationY)
            if (_pathSuccess) and (_pathLength < pathLength):
                pathSuccess = _pathSuccess
                nextX = _nextX
                nextY = _nextY
                pathLength = _pathLength

            # South
            _pathSuccess, _nextX, _nextY, _pathLength = self.findPathNextStep(world, agent.attributes["gridX"], agent.attributes["gridY"], destinationX, destinationY+1)
            if (_pathSuccess) and (_pathLength < pathLength):
                pathSuccess = _pathSuccess
                nextX = _nextX
                nextY = _nextY
                pathLength = _pathLength

            # West
            _pathSuccess, _nextX, _nextY, _pathLength = self.findPathNextStep(world, agent.attributes["gridX"], agent.attributes["gridY"], destinationX-1, destinationY)
            if (_pathSuccess) and (_pathLength < pathLength):
                pathSuccess = _pathSuccess
                nextX = _nextX
                nextY = _nextY
                pathLength = _pathLength


        if (not pathSuccess):
            print("_doNPCAutonavigation: No path found to goal location.  Exiting. (agent: " + agent.name + ")")
            return ActionResult.FAILURE

        if ("doorNeedsToBeClosed" in agent.attributes) and (agent.attributes["doorNeedsToBeClosed"] != None) and (agent.attributes["movesSinceDoorOpen"] == 1):
            # We recently opened a door -- close it
            print("AGENT: CLOSING DOOR")
            doorToClose = agent.attributes["doorNeedsToBeClosed"]
            agent.actionOpenClose(doorToClose, "close")
            agent.attributes["doorNeedsToBeClosed"] = None
            agent.attributes["movesSinceDoorOpen"] = 0
        else:
            # Calculate deltas
            deltaX = nextX - agent.attributes["gridX"]
            deltaY = nextY - agent.attributes["gridY"]

            # First, check to see if we're facing the correct direction.  If not, start rotating in that direction.
            desiredDirection = agent.convertXYDeltasToDirection(deltaX, deltaY)
            if (desiredDirection != agent.attributes["faceDirection"]):
                # We're not facing the correct direction -- rotate
                print("AGENT: ROTATING TO FACE DIRECTION (curDirection: " + agent.attributes["faceDirection"] + ", desiredDirection: " + desiredDirection + ")")
                rotateSuccess = agent.rotateToFaceDirection(desiredDirection)
                print(rotateSuccess)
                return ActionResult.SUCCESS

            # First, check to see if the next step has a barrier (like a door) that needs to be opened
            allObjs = world.getObjectsAt(nextX, nextY)
            # Get a list of objects that are not passable (isPassable == False)
            allObjsNotPassable = [obj for obj in allObjs if (obj.attributes["isPassable"] == False)]

            # If there are no impassable objects, then move the agent one step in the forward direction
            if (len(allObjsNotPassable) == 0):
                # Move agent one step in the forward direction
                #moveSuccess = self.actionMoveAgent(deltaX, deltaY)
                print("AGENT: MOVING FORWARD")
                moveSuccess = agent.actionMoveAgentForwardBackward(direction=+1)
                agent.attributes["movesSinceDoorOpen"] += 1
            else:
                print("AGENT: TRYING TO OPEN IMPASSABLE OBJECT")
                # There's one or more impassable objects -- try to open them.
                for obj in allObjsNotPassable:
                    # Check to see if the object is openable
                    if (obj.attributes["isOpenable"]):
                        # Open the object
                        agent.actionOpenClose(obj, "open")
                        agent.attributes["doorNeedsToBeClosed"] = obj
                        agent.attributes["movesSinceDoorOpen"] = 0
                        # Break out of the loop
                        break

        return ActionResult.SUCCESS

    # Check to see if the agent is directly north, east, south, or west from the destination
    # Returns (True, "north"/"east"/"south"/"west") if the agent is directly adjacent to the destination
    # Returns (False, None) if the agent is not directly adjacent to the destination
    def _isAgentDirectlyAdjacentToDestination(self, queryX, queryY, destinationX, destinationY):
        # Check to see if the locations are identical
        if (queryX == destinationX) and (queryY == destinationY):
            return (True, "identical")

        # Check north
        if (queryX == destinationX) and (queryY == destinationY + 1):
            return (True, "north")
        # Check east
        if (queryX == destinationX + 1) and (queryY == destinationY):
            #return (True, "east")
            return (True, "west")
        # Check south
        if (queryX == destinationX) and (queryY == destinationY - 1):
            return (True, "south")
        # Check west
        if (queryX == destinationX - 1) and (queryY == destinationY):
            #return (True, "west")
            return (True, "east")

        # Default return
        return (False, None)




# Storage class for autopilot action
class AutopilotAction():
    # Constructor
    # ActionType is an enumeration (AutopilotActionType)
    # actionArguments are specific to each type of action
    def __init__(self, actionType, actionArguments:dict, priority=0):
        self.actionType = actionType
        self.args = actionArguments
        self.args['priority'] = priority      # Priority of this action (higher priority actions will be executed first)

    # Priority property getter
    @property
    def priority(self):
        return self.args.get('priority', 0)

    # Priority property setter
    @priority.setter
    def priority(self, value):
        self.args['priority'] = value

    # Timestamp
    @property
    def timestamp(self):
        return self.args.get('timestamp', 0)

    # Timestamp property setter
    @timestamp.setter
    def timestamp(self, value):
        self.args['timestamp'] = value

    # String representation
    def __str__(self):
        return "AutopilotAction: " + str(self.actionType) + " (args: " + str(self.args) + ", priority: " + str(self.priority) + ")"


# Specific action types
class AutopilotAction_GotoXY(AutopilotAction):
    # Constructor
    # Final direction must be one of "north", "east", "south", or "west"
    def __init__(self, x, y, goBesideAndFace:bool = False, finalDirection=None, priority=2):
        self.actionType = AutopilotActionType.GOTO_XY
        self.args = {}
        self.args['destinationX'] = x
        self.args['destinationY'] = y
        self.args['goBesideAndFace'] = goBesideAndFace      # Should we go to the location, or just beside the location and face in its direction?
        self.args['finalDirection'] = finalDirection
        self.args['priority'] = priority

class AutopilotAction_PickupObj(AutopilotAction):
    # Constructor
    def __init__(self, objectToPickUp, callback=None, callbackArgs=None, priority=4):
        self.actionType = AutopilotActionType.PICKUP_OBJ
        self.args = {}
        self.args['objectToPickUp'] = objectToPickUp
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs

class AutopilotAction_PlaceObjInContainer(AutopilotAction):
    # Constructor
    def __init__(self, objectToPlace, container, objectNamesOrTypes=None, containerNamesOrTypes = None, callback=None, callbackArgs=None, priority=3):
        self.actionType = AutopilotActionType.PLACE_OBJ_IN_CONTAINER
        self.args = {}
        self.args['objectToPlace'] = objectToPlace
        self.args['container'] = container
        self.args['objectNamesOrTypes'] = objectNamesOrTypes
        self.args['containerNamesOrTypes'] = containerNamesOrTypes
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs

class AutopilotAction_DropObjAtLocation(AutopilotAction):
    # Constructor
    def __init__(self, objectToPlace, dropX, dropY, objectNamesOrTypes=None, callback=None, callbackArgs=None, priority=3):
        self.actionType = AutopilotActionType.DROP_OBJ_AT_LOCATION
        self.args = {}
        self.args['objectToPlace'] = objectToPlace
        self.args['objectNamesOrTypes'] = objectNamesOrTypes
        self.args['dropLocation'] = [dropX, dropY]
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs

class AutopilotAction_PickupObjectsInArea(AutopilotAction):
    # Constructor
    def __init__(self, x, y, width, height, objectTypes:list, container, excludeObjectsOnAgent=True, maxToTake=-1, callback=None, priority=4):
        self.actionType = AutopilotActionType.FIND_OBJS_AREA_PLACE
        self.args = {}
        self.args['x'] = x
        self.args['y'] = y
        self.args['width'] = width
        self.args['height'] = height
        self.args['objectTypes'] = objectTypes
        self.args['container'] = container
        self.args['excludeObjectsOnAgent'] = excludeObjectsOnAgent      # If True, will not look for objects on the agent if/when the agent is in the area
        self.args['maxToTake'] = maxToTake                              # The maximum number of objects meeting the criteria to take
        self.args['numTaken'] = 0
        self.args['priority'] = priority


class AutopilotAction_LocateBlankTileInArea(AutopilotAction):
    # Constructor
    # If 'allowedContentTypes' is None, then any object type is allowed
    def __init__(self, x, y, width, height, objectTypes:list, allowedContentTypes:list, priority=4):
        self.actionType = AutopilotActionType.LOCATE_BLANK_TILE_IN_AREA
        self.args = {}
        self.args['x'] = x
        self.args['y'] = y
        self.args['width'] = width
        self.args['height'] = height
        self.args['objectTypes'] = objectTypes
        self.args['allowedContentTypes'] = allowedContentTypes
        self.args['numFound'] = 0
        self.args['priority'] = priority

class AutopilotAction_EatObjectInInventory(AutopilotAction):
    # Constructor
    def __init__(self, objectNamesOrTypesToEat, callback=None, callbackArgs=None, priority=3):
        self.actionType = AutopilotActionType.EAT_OBJ_IN_INVENTORY
        self.args = {}
        self.args['objectNamesOrTypesToEat'] = objectNamesOrTypesToEat
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs

    # def runDigInFrontOfAgent(self, args:dict, agent, world):
    #     objectNamesOrTypesToDig = args['objectNamesOrTypesToDig']
class AutopilotAction_DigInFrontOfAgent(AutopilotAction):
    # Constructor
    def __init__(self, objectNamesOrTypesToDig, callback=None, callbackArgs=None, priority=3):
        self.actionType = AutopilotActionType.DIG_IN_FRONT_OF_AGENT
        self.args = {}
        self.args['objectNamesOrTypesToDig'] = objectNamesOrTypesToDig
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs


class AutoPilotAction_BuryInFrontOfAgent(AutopilotAction):
    # Constructor
    def __init__(self, objectNamesOrTypesToDig, objectNamesOrTypesToBury, callback=None, callbackArgs=None, priority=3):
        self.actionType = AutopilotActionType.BURY_IN_FRONT_OF_AGENT
        self.args = {}
        self.args['objectNamesOrTypesToDig'] = objectNamesOrTypesToDig
        self.args['objectNamesOrTypesToBury'] = objectNamesOrTypesToBury
        self.args['priority'] = priority
        self.args['callback'] = callback
        self.args['callbackArgs'] = callbackArgs


class AutopilotAction_Wander(AutopilotAction):
    # Constructor
    def __init__(self, preferredX, preferredY, priority=0):
        self.actionType = AutopilotActionType.WANDER
        self.args = {}
        self.args['preferredX'] = preferredX
        self.args['preferredY'] = preferredY
        self.args['priority'] = priority

class AutopilotAction_WanderSlow(AutopilotAction):
    # Constructor
    def __init__(self, preferredX, preferredY, priority=0):
        self.actionType = AutopilotActionType.WANDER_SLOW
        self.args = {}
        self.args['preferredX'] = preferredX
        self.args['preferredY'] = preferredY
        self.args['priority'] = priority

class AutopilotAction_Wait(AutopilotAction):
    # Constructor
    def __init__(self, priority=0):
        self.actionType = AutopilotActionType.WAIT
        self.args = {}
        self.args['priority'] = priority

class AutopilotAction_PostDiscoveryFeedUpdate(AutopilotAction):
    # Constructor
    def __init__(self, contentStr:str, signals:list=[], priority=0):
        self.actionType = AutopilotActionType.POST_DISCOVERY_FEED_UPDATE
        self.args = {}
        self.args['contentStr'] = contentStr
        self.args['signals'] = signals
        self.args['priority'] = priority

class AutopilotAction_MoveRelative(AutopilotAction):
    # Constructor
    def __init__(self, direction, priority=0):
        self.actionType = AutopilotActionType.MOVE_RELATIVE
        self.args = {}
        self.args['direction'] = direction      # Direction should be "N", "E", "S", or "W"
        self.args['priority'] = priority

class AutopilotAction_RotateToFaceDirection(AutopilotAction):
    # Constructor
    def __init__(self, direction, priority=0):
        self.actionType = AutopilotActionType.ROTATE_TO_FACE_DIRECTION
        self.args = {}
        self.args['direction'] = direction      # Direction should be "N", "E", "S", or "W"
        self.args['priority'] = priority

class AutopilotAction_CheckCondition(AutopilotAction):
    # Constructor
    def __init__(self, conditionCallback, addActionIfTrue=None, addActionIfFalse=None, priority=0):
        self.actionType = AutopilotActionType.CHECK_CONDITION
        self.args = {}
        self.args['conditionCallback'] = conditionCallback
        self.args['addActionIfTrue'] = addActionIfTrue
        self.args['addActionIfFalse'] = addActionIfFalse
        self.args['priority'] = priority


# Enumeration for types of autopilot actions
class AutopilotActionType(Enum):
    GOTO_XY                 = 0
    PICKUP_OBJ              = 1
    PLACE_OBJ_IN_CONTAINER  = 2
    WANDER                  = 3
    WAIT                    = 4
    FIND_OBJS_AREA_PLACE    = 5
    EAT_OBJ_IN_INVENTORY    = 6
    LOCATE_BLANK_TILE_IN_AREA = 7
    DIG_IN_FRONT_OF_AGENT   = 8
    BURY_IN_FRONT_OF_AGENT  = 9
    DROP_OBJ_AT_LOCATION    = 10
    POST_DISCOVERY_FEED_UPDATE = 11
    MOVE_RELATIVE           = 12
    ROTATE_TO_FACE_DIRECTION = 13
    CHECK_CONDITION         = 14
    WANDER_SLOW             = 15
