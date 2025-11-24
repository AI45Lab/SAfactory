# KnowledgeScorer.py

import json
from enum import Enum

# # Enumeration for layer types
# class ActionType(Enum):
#     MOVE_FORWARD    = 0
#     MOVE_BACKWARD   = 1
#     ROTATE_CW       = 2
#     ROTATE_CCW      = 3
#     PICKUP          = 4
#     PUT             = 5
#     DROP            = 6
#     OPEN            = 7
#     CLOSE           = 8
#     ACTIVATE        = 9
#     DEACTIVATE      = 10
#     TALK            = 11
#     EAT             = 12
#     READ            = 13
#     USE             = 14

# Enumeration for property operators
class PropertyOperator(Enum):
    EQUALS = 0
    LESS_THAN = 1
    GREATER_THAN = 2
    LESS_THAN_OR_EQUAL = 3
    GREATER_THAN_OR_EQUAL = 4
    NOT_EQUAL = 5
    CONTAINS_LIST = 6                   # e.g. ["red", "green", "blue"]


# Storage class for an object reference, which may or may not include additional property filtering/scoping.
class ObjectReference():
    def __init__(self, dictIn:str): # objectUUID:str, objectName:str, objectType:str, properties:dict):
        # Keep track of any errors
        self.errors = []

        # Can refer to a specific object (by UUID), an object by name ('name'), or a class of objects (by 'type')
        self.objectUUID = None
        self.objectName = None
        self.objectType = None

        # Parse object UUID, name, or type from the dictionary (but, should have only one)
        refCount = 0
        if ('objectUUID' in dictIn):
            self.objectUUID = dictIn['objectUUID']
            # The type of the UUID should be an int, not a string.
            if (type(self.objectUUID) is not int):
                self.errors.append("Object UUID must be an integer")

            refCount += 1
        if ('objectName' in dictIn):
            self.objectName = dictIn['objectName']
            refCount += 1
        if ('objectType' in dictIn):
            self.objectType = dictIn['objectType']
            refCount += 1
        # Check that only one of these is set
        if (refCount > 1):
            self.errors.append("Object reference can only have one of 'objectUUID', 'objectName', or 'objectType'")
        elif (refCount == 0):
            self.errors.append("Object reference must have one of 'objectUUID', 'objectName', or 'objectType'")


        # Parse properties (which are called 'scope' in the JSON dictionary)
        self.properties = {}
        if ('scope' in dictIn):
            propListToParse = dictIn['scope']
            for propToParse in propListToParse:
                prop = ObjectProperty(propToParse)
                if (prop.isValid):
                    self.properties[prop.propertyName] = prop
                else:
                    self.errors.append("Unable to parse property: " + str(propToParse))

        # Validate (TODO, make this more robust)
        self.isValid = True if (len(self.errors) == 0) else False


    # Get the list of object(s) meeting this criteria from a given step in the world history
    def findInWorldHistoryStep(self, worldHistoryStep):
        # Keys: 'step', 'sizeX', 'sizeY', 'grid'.  Grid is a 2D array, with each element being a list of objects at that location
        out = []

        # Iterate through the 2D grid
        for x in range(worldHistoryStep['sizeX']):
            for y in range(worldHistoryStep['sizeY']):
                # Get the list of objects at this location
                objList = worldHistoryStep['grid'][x][y]
                # Filter the list of objects by the criteria in this object reference
                objListFiltered = self._filterObjsByCriteria(objList)
                # Add the filtered list to the output
                out.extend(objListFiltered)

        return out

    # Get any objects meeting the criteria defined in this object, from a list of input objects
    def _filterObjsByCriteria(self, objListIn):
        out = []

        # Step 1: Initial filtering: The object can be referenced by either UUID, name, or type
        candidateObjects = []
        if (self.objectUUID is not None):
            # Find the object with this UUID
            for obj in objListIn:
                if (obj['uuid'] == self.objectUUID):
                    candidateObjects.append(obj)
        elif (self.objectName is not None):
            # Find the object(s) with this name
            nameSanitized = self.objectName.lower().strip()
            for obj in objListIn:
                if (obj['name'].lower().strip() == nameSanitized):
                    candidateObjects.append(obj)
        elif (self.objectType is not None):
            # Find the object(s) with this type
            typeSanitized = self.objectType.lower().strip()
            for obj in objListIn:
                if (obj['type'].lower().strip() == typeSanitized):
                    candidateObjects.append(obj)


        # Step 2: Now filter the candidate list by any properties that were supplied in the scope
        # If there are no scoping properties, then we're done
        if (len(self.properties) == 0):
            return candidateObjects

        # Iterate through the candidate objects, and only keep the ones that meet the criteria
        for obj in candidateObjects:
            meetsCriteria = True
            for property in self.properties:
                result = property.checkObjectMeetsPropertyCriteria(obj)
                if (result == False):
                    # At least one of the property criteria doesn't match, so filter out this object
                    meetsCriteria = False
                    break

            # If we reach here and the object meets all criteria, then add it to the output list
            if (meetsCriteria):
                out.append(obj)

        return out


    # String method
    def __str__(self):
        strOut = "ObjectReference("
        if (self.objectUUID is not None):
            strOut += "UUID: " + str(self.objectUUID) + ", "
        if (self.objectName is not None):
            strOut += "Name: " + self.objectName + ", "
        if (self.objectType is not None):
            strOut += "Type: " + self.objectType + ", "
        if (len(self.properties) > 0):
            strOut += "Properties: "
            for propName in sorted(self.properties.keys()):
                strOut += "'" + propName + "': " + str(self.properties[propName]) + ", "
        strOut += ")"
        return strOut


# Storage class for an object-operator-property combination (e.g. 'color', 'equals', 'red')
class ObjectProperty():
    def __init__(self, dictIn:dict):
        # Keep track of any errors
        self.errors = []

        # Parse
        self.propertyName = _parseDictValue(dictIn, 'propertyName', self.errors)
        propertyOperatorStr = _parseDictValue(dictIn, 'propertyOperator', self.errors)
        self.propertyOperator = _parsePropertyOperator(propertyOperatorStr, self.errors)
        self.propertyValue = _parseDictValue(dictIn, 'propertyValue', self.errors)

        # Secondary checks
        # If the operator is CONTAINS_LIST, then the property value must be a list
        if (self.propertyOperator == PropertyOperator.CONTAINS_LIST):
            if (type(self.propertyValue) is not list):
                self.errors.append("Property operator is CONTAINS_LIST, but property value is not a list (instead it is '" + str(type(self.propertyValue)) + "')")

        # Validate (TODO, make this more robust)
        self.isValid = True if (len(self.errors) == 0) else False


    # Check if a specific object meets the criteria defined in this property
    def checkObjectMeetsPropertyCriteria(self, objIn:object):
        # Check if the object has this property
        if (self.propertyName not in objIn['attributes']):
            # The object doesn't have this property, so it can't meet the criteria
            return False

        # Get the property value from the object
        propValue = objIn['attributes'][self.propertyName]

        # Check if the property value matches the criteria, based on the operator
        if (self.propertyOperator == PropertyOperator.EQUALS):
            if (propValue == self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.LESS_THAN):
            if (propValue < self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.GREATER_THAN):
            if (propValue > self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.LESS_THAN_OR_EQUAL):
            if (propValue <= self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.GREATER_THAN_OR_EQUAL):
            if (propValue >= self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.NOT_EQUAL):
            if (propValue != self.propertyValue):
                return True
            else:
                return False
        elif (self.propertyOperator == PropertyOperator.CONTAINS_LIST):
            # To meet this criteria, the property value must be one of the elements in the list.
            if (propValue in self.propertyValue):
                return True
            else:
                return False

        # If we reach here, then there is an unknown operator
        print("ERROR: ObjectProperty.checkObjectMeetsProperty(): Unknown operator (" + str(self.propertyOperator) + ")")
        return False



    # String method
    def __str__(self):
        return "(" + str(self.propertyName) + ", " + str(self.propertyOperator) + ", " + str(self.propertyValue) + ")"





#
#   Storage classes for specific kinds of knowledge
#

# Storage class for a measurement, which contains an object reference, and a property
class Measurement():
    def __init__(self, dictIn:dict, step:int=None):
        # Keep track of any errors
        self.errors = []

        # Set the step that this knowledge was generated at
        self.step = step

        # Set the score
        self.score = None
        self.scoreJustification = []

        # Parse

        # Field 1: Object reference
        objReference = _parseDictValue(dictIn, 'object', self.errors)
        self.objectReference = ObjectReference(objReference)
        self.errors += self.objectReference.errors

        # Field 2: Property
        property = _parseDictValue(dictIn, 'property', self.errors)
        self.propertyMeasurement = ObjectProperty(property)
        self.errors += self.propertyMeasurement.errors

        # Validate (TODO, make this more robust)
        self.isValid = True if (len(self.errors) == 0) else False


    # Score whether this measurement is true or not, based on the objects in the world history step
    def evaluate(self, worldHistoryStep:list):
        # Clear the score and justification
        self.score = None
        self.scoreJustification = []

        # First, filter the objectsIn to only those that match the object reference
        objectsMatchingObjectReference = self.objectReference.findInWorldHistoryStep(worldHistoryStep)
        # If the list is empty, then the measurement is false
        if (len(objectsMatchingObjectReference) == 0):
            self.score = 0
            self.scoreJustification.append("Could not find any objects in the world at step (" + str(self.step) + ") that match the object reference uuid/name/type, and/or scope criteria.")
            return self.score

        # Next, filter the objects to only those that match the property
        objectsMatchingProperty = []
        for obj in objectsMatchingObjectReference:
            if (self.propertyMeasurement.checkObjectMeetsPropertyCriteria(obj)):
                objectsMatchingProperty.append(obj)

        # Set the score to be the proportion of objects that match the property, out of the objects that match the object reference
        self.score = len(objectsMatchingProperty) / len(objectsMatchingObjectReference)
        self.scoreJustification.append("Found " + str(len(objectsMatchingProperty)) + " objects that match the property criteria, out of " + str(len(objectsMatchingObjectReference)) + " objects that match the object reference criteria (total proportion: " + str(self.score) + ")")
        return self.score


    # String method
    def __str__(self):
        return "Measurement(" + str(self.objectReference) + ", PropertyMeasurement: " + str(self.propertyMeasurement) + "), Step: " + str(self.step) + ")"



#
#   Helper functions
#
def _parseDictValue(dictIn:dict, fieldName:str, errors:list):
    if (fieldName in dictIn):
        return dictIn[fieldName]
    else:
        errors.append("Missing field in property: " + fieldName)
        return None

# Convert form a string representation of the Enum to the Enum
def _parsePropertyOperator(strIn:str, errors:list):
    if (strIn == None):
        errors.append("Missing property operator")
        return None

    # Sanitize the input
    strIn = strIn.strip().upper()

    if (strIn == "EQUALS"):
        return PropertyOperator.EQUALS
    elif (strIn == "LESS_THAN"):
        return PropertyOperator.LESS_THAN
    elif (strIn == "GREATER_THAN"):
        return PropertyOperator.GREATER_THAN
    elif (strIn == "LESS_THAN_OR_EQUAL"):
        return PropertyOperator.LESS_THAN_OR_EQUAL
    elif (strIn == "GREATER_THAN_OR_EQUAL"):
        return PropertyOperator.GREATER_THAN_OR_EQUAL
    elif (strIn == "NOT_EQUAL"):
        return PropertyOperator.NOT_EQUAL
    elif (strIn == "CONTAINS_LIST"):
        return PropertyOperator.CONTAINS_LIST
    else:
        #raise ValueError("Unknown property operator: " + strIn)
        errors.append ("Unknown property operator: " + strIn)
        return None



#
#   Knowledge scorer
#
class KnowledgeScorer:
    # Constructor
    def __init__(self, world):

        # Different types of scientific knowledge
        self.measurements = []
        self.hypotheses = []
        self.hypothesisConfirmReject = []
        self.laws = []
        self.theories = []
        self.models = []

        self.world = world


    # Add knowledge: a measurement
    def addMeasurement(self, measurement:Measurement):
        self.measurements.append(measurement)


    # Evaluate whether a measurement is correct or incorrect, based on the world history.
    def evaluateMeasurement(self, measurement:Measurement):
        # Step 0: First, make sure that the measurement is valid (and has no errors)
        if (not measurement.isValid):
            return None

        # Step 1: Get the step that the measurement is referencing
        measurementStep = measurement.step

        # Step 2: Get the world state at that step
        worldState = self.world.getWorldHistoryAtStep(measurementStep)
        if (worldState == None):
            return None

        # Perform the scoring, return the score
        return measurement.evaluate(worldState)




#
#   Tests
#

if __name__ == "__main__":
    # Test ObjectReference
    print("Testing ObjectReference")

    # JSON string
    #jsonStr = """{"objectUUID": "1234"}"""
    jsonStr = """{"objectUUID": "1234", "scope":[{"propertyName":"color", "propertyOperator":"equals", "propertyValue":"red"}, {"propertyName":"size", "propertyOperator":"less_than", "propertyValue":5}]}"""
    jsonStr = """
    {
        "object": {"objectUUID": "1234", "scope":[{"propertyName":"color", "propertyOperator":"equals", "propertyValue":"red"}, {"propertyName":"size", "propertyOperator":"less_than", "propertyValue":5}]},
        "property": {"propertyName":"color", "propertyOperator":"equals", "propertyValue":"blue"}
    }
        """
    # Convert to a dictionary
    dictIn = json.loads(jsonStr)

    # Create an ObjectReference
    #objRef = ObjectReference(dictIn)
    #print(objRef.errors)
    #print(objRef)

    # Create a Measurement
    measurement = Measurement(dictIn)
    print(measurement.errors)
    print(measurement)