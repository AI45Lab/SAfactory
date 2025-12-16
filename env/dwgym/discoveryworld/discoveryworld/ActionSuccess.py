# ActionSuccess.py
from enum import Enum

# Enumeration to define different types of action results (success, completion, failure)
class ActionResult(Enum):
    SUCCESS     = 0
    COMPLETED   = 1
    SUSPEND     = 2
    FAILURE     = 3
    INVALID     = 4


# Enumeration to define message importance
class MessageImportance(Enum):
    LOW         = 0
    NORMAL      = 1
    HIGH        = 2



# Storage class for the result of an action
class ActionSuccess:
    # Constructor
    def __init__(self, success, message, importance = MessageImportance.NORMAL):
        self.success = success
        self.message = message
        self.importance = importance
        self.data = {}

    # To a dictionary
    def toPartialDict(self):
        out = {
            "success": self.success,
            "message": self.message,
            "importance": self.importance.value,
            # Note, omits the data dictionary because it may not be easily serializable
        }
        return out

    # String method
    def __str__(self):
        return "ActionSuccess(" + str(self.success) + ", " + str(self.importance) + ", " + str(self.message) + ")"




# Storage class for the result of a "use with" action
class UseWithSuccess(ActionSuccess):
    # Constructor
    def __init__(self, success, message, generatedItems = [], importance = MessageImportance.NORMAL):
        ActionSuccess.__init__(self, success, message, importance)
        self.data['generatedItems'] = generatedItems



    # String method
    def __str__(self):
        return "UseWithSuccess(" + str(self.success) + ", " + str(self.importance) + ", " + str(self.message) + ", " + str(self.data['generatedItems']) + ")"


# Storage class for the result of a dialog action
class DialogSuccess(ActionSuccess):
    # Constructor
    def __init__(self, success, message, dialogOptions = [], importance = MessageImportance.NORMAL):
        ActionSuccess.__init__(self, success, message, importance)
        self.data['dialogOptions'] = dialogOptions

    # String method
    def __str__(self):
        return "DialogSuccess(" + str(self.success) + ", " + str(self.importance) + ", " + str(self.message) + ", " + str(self.data['dialogOptions']) + ")"