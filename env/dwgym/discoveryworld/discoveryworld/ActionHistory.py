# ActionHistory.py

from enum import Enum

from discoveryworld.ActionSuccess import ActionSuccess
from discoveryworld.objects import Object

# Enumeration for layer types
class ActionType(Enum):
    MOVE_FORWARD    = 0
    MOVE_BACKWARD   = 1
    ROTATE_CW       = 2
    ROTATE_CCW      = 3
    PICKUP          = 4
    PUT             = 5
    DROP            = 6
    OPEN            = 7
    CLOSE           = 8
    ACTIVATE        = 9
    DEACTIVATE      = 10
    TALK            = 11
    EAT             = 12
    READ            = 13
    USE             = 14
    DISCOVERY_FEED_GET_UPDATES = 15
    DISCOVERY_FEED_GET_ARTICLES = 16
    DISCOVERY_FEED_GET_POST_BY_ID = 17
    DISCOVERY_FEED_CREATE_UPDATE = 18
    DISCOVERY_FEED_CREATE_ARTICLE = 19
    MOVE_DIRECTION = 20
    ROTATE_DIRECTION = 21
    TELEPORT_TO_LOCATION = 22
    TELEPORT_TO_OBJECT = 23



# Returns a dictionary of action descriptions
def getActionDescriptions(limited:bool = False):
    actionDescriptions = {
        #ActionType.MOVE_FORWARD.name:   {"args": [], "desc": "move forward 1 step (in whatever direction the agent is facing)"},
        #ActionType.MOVE_BACKWARD.name:  {"args": [], "desc": "move backward 1 step (backwards from whatever direction the agent is facing)"},
        #ActionType.ROTATE_CCW.name:     {"args": [], "desc": "move counter-clockwise 90 degrees"},
        #ActionType.ROTATE_CW.name:      {"args": [], "desc": "move clockwise 90 degrees"},
        ActionType.PICKUP.name:         {"args": ["arg1"], "desc": "pick up an object (arg1)"},
        ActionType.DROP.name:           {"args": ["arg1"], "desc": "drop an object (arg1)"},
        ActionType.PUT.name:            {"args": ["arg1", "arg2"], "desc": "put an object (arg1) in/on another object (arg2), or give an object (arg1) to another agent (arg2)"},
        ActionType.OPEN.name:           {"args": ["arg1"], "desc": "open an object (arg1)"},
        ActionType.CLOSE.name:          {"args": ["arg1"], "desc": "close an object (arg1)"},
        ActionType.ACTIVATE.name:       {"args": ["arg1"], "desc": "activate an object (arg1)"},
        ActionType.DEACTIVATE.name:     {"args": ["arg1"], "desc": "deactivate an object (arg1)"},
        ActionType.TALK.name:           {"args": ["arg1"], "desc": "talk to another agent (arg1)"},
        ActionType.EAT.name:            {"args": ["arg1"], "desc": "eat an object (arg1)"},
        ActionType.READ.name:           {"args": ["arg1"], "desc": "read an object (arg1)"},
        ActionType.USE.name:            {"args": ["arg1", "arg2"], "desc": "use an object (arg1), e.g. a thermometer, on another object (arg2), e.g. water."},

        ActionType.MOVE_DIRECTION.name:     {"args": ["arg1"], "desc": "move in a specific direction (arg1), which is one of 'north', 'east', 'south', or 'west'."},
        ActionType.ROTATE_DIRECTION.name:   {"args": ["arg1"], "desc": "rotate to face a specific direction (arg1), which is one of 'north', 'east', 'south', or 'west'."},
        ActionType.TELEPORT_TO_LOCATION.name:   {"args": ["arg1"], "desc": "teleport to a specific location (arg1), by name. A list of valid teleport locations is provided elsewhere."},
        ActionType.TELEPORT_TO_OBJECT.name:     {"args": ["arg1"], "desc": "teleport beside a specific object (arg1). 'arg1' should be the UUID of the object to teleport to."},

        ActionType.DISCOVERY_FEED_GET_UPDATES.name:     {"args": [], "desc": "read the latest status updates on discovery feed"},
        #ActionType.DISCOVERY_FEED_GET_ARTICLES.name:    {"args": [], "desc": "read the latest scientific articles on discovery feed"},
        ActionType.DISCOVERY_FEED_GET_POST_BY_ID.name:  {"args": ["arg1"], "desc": "read a specific post on discovery feed (arg1). 'arg1' should be the integer ID of the post."},
        #ActionType.DISCOVERY_FEED_CREATE_UPDATE.name:   {"args": ["arg1"], "desc": "create a status update on discovery feed (arg1)"},
        #ActionType.DISCOVERY_FEED_CREATE_ARTICLE.name:  {"args": ["arg1"], "desc": "create a scientific article on discovery feed (arg1)"}
    }

    # Limited mode allows removing some actions that may be challenging for some agent models
    if (limited):
        # Remove the Forward/backward/rotate ccw/rotate cw actions
        # NOTE: THIS IS A TEMPORARY TEST!
        #actionDescriptions.pop(ActionType.MOVE_FORWARD.name)
        #actionDescriptions.pop(ActionType.MOVE_BACKWARD.name)
        #actionDescriptions.pop(ActionType.ROTATE_CCW.name)
        #actionDescriptions.pop(ActionType.ROTATE_CW.name)

        # Remove the talk action
        actionDescriptions.pop(ActionType.TALK.name)
        # Remove the discovery feed actions
        actionDescriptions.pop(ActionType.DISCOVERY_FEED_GET_UPDATES.name)
        actionDescriptions.pop(ActionType.DISCOVERY_FEED_GET_ARTICLES.name)
        actionDescriptions.pop(ActionType.DISCOVERY_FEED_GET_POST_BY_ID.name)
        actionDescriptions.pop(ActionType.DISCOVERY_FEED_CREATE_UPDATE.name)
        actionDescriptions.pop(ActionType.DISCOVERY_FEED_CREATE_ARTICLE.name)

    return actionDescriptions



# Stores the action history for one agent
class ActionHistory:
    # Constructor
    def __init__(self, world):
        self.history = []
        self.world = world

    # Add an action to the history
    def add(self, actionType:ActionType, arg1, arg2, result:ActionSuccess):
        arg1UUID = None
        arg2UUID = None
        if (arg1 != None):
            if (isinstance(arg1, Object)):
                arg1UUID = arg1.uuid
        if (arg2 != None):
            if (isinstance(arg2, Object)):
                arg2UUID = arg2.uuid

        packed = {
            'actionType': actionType,
            'arg1': arg1,
            'arg1UUID': arg1UUID,
            'arg2': arg2,
            'arg2UUID': arg2UUID,
            'success': result.success,
            'message': result.message,
            'step': self.world.getStepCounter(),
            'result': result
        }

        return self.history.append(packed)

    def getLastAction(self):
        if (len(self.history) > 0):
            return self.history[-1]
        else:
            return None

    # Only returns the last action if it happened on the last step.  Otherwise, None
    def getLastStepAction(self):
        if (len(self.history) > 0):
            if (self.history[-1]['step'] == self.world.getStepCounter()):
                return self.history[-1]
        return None

    # Query for a specific action
    # Queries for specific actions, but arg1 and arg2 should be objects (or None)
    # If arg1 or arg2 are objects, then the query will be for the UUID of the object
    # If arg1 or arg2 are '*', then the query will be for any object
    def queryActionObjects(self, actionType:ActionType, arg1=None, arg2=None, stopAtFirst:bool = False):
        out = []

        # Find the UUIDs of the objects
        arg1Query = arg1
        arg2Query = arg2
        if isinstance(arg1, Object):
            arg1Query = arg1.uuid

        if isinstance(arg2, Object):
            arg2Query = arg2.uuid

        # Search for the query action
        #print("Query: " + str(actionType) + " " + str(arg1Query) + " " + str(arg2Query))
        for action in self.history:
            #print("Action: " + str(action['actionType']) + " " + str(action['arg1UUID']) + " " + str(action['arg2UUID']))
            if (action['actionType'] == actionType
                and (action['arg1UUID'] == arg1Query or arg1Query == "*")
                and (action['arg2UUID'] == arg2Query or arg2Query == "*")
            ):
                out.append(action)
                #print("FOUND!!!!")
                if stopAtFirst:
                    break

        # Return the list of matching actions
        return out

    # Very specific function, used to see if instruments have been used on specific Arg2s
    def queryActionObjectsByArg2Type(self, actionType:ActionType, arg1=None, arg2TypeContains="", stopAtFirst:bool = False):
        out = []

        # Find the UUIDs of the objects
        arg1Query = arg1
        if isinstance(arg1, Object):
            arg1Query = arg1.uuid

        # Search for the query action
        #print("Query: " + str(actionType) + " " + str(arg1Query) + " " + str(arg2Query))
        for action in self.history:
            #print("Action: " + str(action['actionType']) + " " + str(action['arg1UUID']) + " " + str(action['arg2UUID']))
            if (action['actionType'] == actionType
                and (action['arg1UUID'] == arg1Query or arg1Query == "*")
                and (arg2TypeContains in action['arg2'].type)
            ):
                out.append(action)
                #print("FOUND!!!!")
                if stopAtFirst:
                    break

        # Return the list of matching actions
        return out

    # Very specific function, used to see if actions have been taken on specific Arg1s
    def queryActionObjectsByArg1Type(self, actionType:ActionType, arg1TypeContains="", stopAtFirst:bool = False):
        out = []

        # Search for the query action
        #print("Query: " + str(actionType) + " " + str(arg1Query) + " " + str(arg2Query))
        for action in self.history:
            #print("Action: " + str(action['actionType']) + " " + str(action['arg1UUID']) + " " + str(action['arg2UUID']))
            if (action['actionType'] == actionType
                and (arg1TypeContains in action['arg1'].type)
            ):
                out.append(action)
                #print("FOUND!!!!")
                if stopAtFirst:
                    break

        # Return the list of matching actions
        return out


    # Export the action history to a list that can be converted to JSON
    def exportToJSONAbleList(self):
        out = []
        for action in self.history:
            packed = {
                'actionType': action['actionType'].name,
                'arg1': None,
                'arg2': None,
                'success': action['success'],
                'message': action['message'],
                'step': action['step']
                #'result': not saved here
            }
            if (action['arg1'] != None):
                # Check type -- if it's an Object, then we need to pack it differently
                if (isinstance(action['arg1'], Object)):
                    packed['arg1'] = {"objUUID": action['arg1'].uuid}
                else:
                    # It's not an Object, so just pack it directly
                    packed['arg1'] = action['arg1']
                #packed['arg1'] = {"objUUID": action['arg1'].uuid}
            if (action['arg2'] != None):
                # Check type -- if it's an Object, then we need to pack it differently
                if (isinstance(action['arg2'], Object)):
                    packed['arg2'] = {"objUUID": action['arg2'].uuid}
                else:
                    # It's not an Object, so just pack it directly
                    packed['arg2'] = action['arg2']
                #packed['arg2'] = {"objUUID": action['arg2'].uuid}

            out.append(packed)

        return out


    # Export the action history to a list that can be converted to JSON
    def exportToJSONAbleListHumanReadable(self, lastNSteps:int = 0):
        out = []

        # Determine what steps we should export from.  If lastNSteps is 0, then export all steps
        currentWorldStep = self.world.getStepCounter()
        exportAboveStep = currentWorldStep - lastNSteps
        if (lastNSteps <= 0):
            exportAboveStep = -1

        for action in self.history:
            # If we're only exporting the last N steps, then skip any actions that happened before that
            step = action['step']
            if (step <= exportAboveStep):
                continue

            # Pack the action
            packed = {
                'actionType': action['actionType'].name,
                'arg1': None,
                'arg2': None,
                'success': action['success'],
                'message': action['message'],
                'step': action['step']
                #'result': not saved here
            }
            if (action['arg1'] != None):
                # Check type -- if it's an Object, then we need to pack it differently
                if (isinstance(action['arg1'], Object)):
                    #packed['arg1'] = {"objUUID": action['arg1'].uuid}
                    packed['arg1_desc'] = action['arg1'].name + ": " + action['arg1'].getTextDescription()
                    packed['arg1'] = action['arg1'].uuid
                else:
                    # It's not an Object, so just pack it directly
                    packed['arg1'] = action['arg1']
            else:
                # It is None, so delete this key from packed
                del packed['arg1']

            if (action['arg2'] != None):
                # Check type -- if it's an Object, then we need to pack it differently
                if (isinstance(action['arg2'], Object)):
                    #packed['arg2'] = {"objUUID": action['arg2'].uuid}
                    packed['arg2_desc'] = action['arg2'].name + ": " + action['arg2'].getTextDescription()
                    packed['arg2'] = action['arg2'].uuid
                else:
                    # It's not an Object, so just pack it directly
                    packed['arg2'] = action['arg2']
            else:
                # It is None, so delete this key from packed
                del packed['arg2']
                    #packed['arg2'] = action['arg2']
                #packed['arg2'] = {"objUUID": action['arg2'].uuid}

            out.append(packed)

        return out


    def __str__(self):
        strOut = ""
        for action in self.history:
            strOut += str(action) + "\n"
        return strOut
