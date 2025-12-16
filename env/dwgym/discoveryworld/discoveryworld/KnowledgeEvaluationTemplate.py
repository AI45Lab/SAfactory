# KnowledgeEvaluationTemplate.py

import argparse
import pickle
import zlib
import json


#
#   Storage class for an object
#
class KEObject():
    # Constructor
    def __init__(self, initDict:dict):
        self.uuid = initDict['uuid']
        self.name = initDict['name']
        self.type = initDict['type']
        self.contents = initDict['contents']
        self.parts = initDict['parts']
        self.attributes = initDict['attributes']
        self.actionHistory = initDict['actionHistory']

    # Override __getattr__ to allow any 'attribute' to be accessed as a direct attribute. 
    # If the attribute does not exist, return None
    def __getattr__(self, item):
        # Check if the attribute exists in self.attributes
        value = self.attributes.get(item, None)
        if (value != None):
            return value

        # Check if the attribute exists in self.materials (which is a list of materials)
        for material in self.materials:
            value = material.get(item, None)
            if (value != None):
                return value

        # Otherwise, return None
        return None

    
    # This sets the specific step in the action history that will be checked with the `tookAction` method
    def setStepToCheck(self, step:int):
        self.actionHistoryStep = step

    # Check to see if (a) this is an agent, and (b) the action provided in the argument is in the agent's action list    
    def tookAction(self, actionTypeStr, arg1=None, arg2=None):        
        # First, check to see if this object has a populated action history
        if (self.actionHistory == None) or (len(self.actionHistory) == 0):
            return False

        #print("Started for object: " + self.name + " (" + str(self.uuid) + ")")

        # Next, check the action history to see if an action was taken at step `self.actionHistoryStep`
        actionToCheck = None
        for action in self.actionHistory:
            if (action['step'] == self.actionHistoryStep):                
                actionToCheck = action
                break

        # If no action was taken at the specified step, return False
        if (actionToCheck == None):
            #print("No action taken at step " + str(self.actionHistoryStep))
            return False
        #print("Action taken at step " + str(self.actionHistoryStep) + ": " + str(actionToCheck))

        # Check the action type for a match
        if (actionTypeStr != actionToCheck['actionType']):
            #print("Wrong action type (" + actionTypeStr + " != " + actionToCheck['actionType'] + ")")
            return False

        # print("Correct action type (" + actionTypeStr + " == " + actionToCheck['actionType'] + ")")
        # print("arg1.uuid: " + str(arg1.uuid))
        # print("arg2: " + str(arg2))
        # print("actionToCheck['arg1'].uuid: " + str(actionToCheck['arg1']['objUUID']))
        # print("actionToCheck['arg2']: " + str(actionToCheck['arg2']))


        # If we reach here, check to see that the arguments match. 
        # First, check for `None`` matches
        if ((arg1 == None) and (actionToCheck['arg1'] != None)) or ((arg1 != None) and (actionToCheck['arg1'] == None)):
            return False
        if ((arg2 == None) and (actionToCheck['arg2'] != None)) or ((arg2 != None) and (actionToCheck['arg2'] == None)):
            return False
        
        # Then, check for object matches
        if ((arg1 != None) and (actionToCheck['arg1'] != None)):
            if (arg1.uuid != actionToCheck['arg1']['objUUID']):
                return False
        if ((arg2 != None) and (actionToCheck['arg2'] != None)):
            if (arg2.uuid != actionToCheck['arg2']['objUUID']):
                return False

        print("FOUND")        
        # If we reach here, all checks have passed -- the action taken at the specified step matches the action provided in the arguments
        return True




#
#   Knowledge Evaluation Template
# 
class KnowledgeEvaluationTemplate():
    # Constructor
    def __init__(self):
        # Step 1: Load the world history
        self.worldHistoryFilename = "sandbox/worldHistory.pickle"
        self.worldHistory = self.loadWorldHistory(self.worldHistoryFilename)

        self.evaluationResults = {
            "conditionMetCount": 0,
            "assertionMetCount": 0,
        }

        self.curStep = 0    # The current step being checked

    #
    #   Import/Export
    #

    # Import the world history from a (pickled) file
    def loadWorldHistory(self, filename):
        print("Importing world history from file: " + filename + "...")
        f = open(filename, "rb")
        worldHistory = pickle.loads(f.read())
        f.close()        
        print("Import complete.")
        return worldHistory

    # Export the results of the evaluation
    def exportEvaluationResults(self, filename):
        print("Exporting evaluation results to file: " + filename + "...")
        f = open(filename, "w")
        f.write(json.dumps(self.evaluationResults))
        f.close()        
        print("Export complete.")

    # Return the length of the world history (in number of steps)
    def getWorldHistoryLength(self):
        return len(self.worldHistory)


    #
    #   World History access functions
    #
    def getWorldHistoryAtStep(self, step):
        # Return the world history at the specified step
        # Since the world history is pickled then compressed, this will require uncompressing then unpickling        
        if (step < 0 or step >= len(self.worldHistory)):
            return None

        # Get compressed history for a given step
        compressedPickled = self.worldHistory[step]

        # Uncompress the history
        pickled = zlib.decompress(compressedPickled)
        # Unpickle the history
        history = pickle.loads(pickled)
                    
        # Convert the history to just a list of all objects at that step
        allObjects = []
        allObjectsByUUID = {}
        # Traverse grid, adding all objects to the list
        grid = history['grid']
        for row in grid:
            for cell in row:                
                for obj in cell:
                    convertedObj = KEObject(obj)
                    # Set the action history step that will be checked, to be the current step
                    convertedObj.setStepToCheck(step)
                    # Add the object to the list, and the UUID-to-object LUT
                    allObjects.append(convertedObj)
                    allObjectsByUUID[convertedObj.uuid] = convertedObj

                    # if (convertedObj.type == "agent"):
                    #     print(obj)
                    #     print("\n\n")

        return allObjects, allObjectsByUUID



    #
    #   Functions for building the hypothesis queries
    #

    def getProperty(self, obj, property):
        # Return the value of the property of the object

        pass


    #
    #   Functions for testing whether the hypothesis assertion (i.e. it's THEN statement) is true    
    #
    def testHypothesisAssertion(self, hypothesisAssertion):
        pass

    #
    #   The hypothesis itself
    #
    def testHypothesis(self):
        pass


    def exampleHypothesis1(self, obj1):
        # If a mushroom exists, and it is either red or pink, then it is poisonous
        def exampleAssertion(self, obj1):
            # Check that 'obj1' has 'isPoisonous' = true            
            if (obj1.color == "pink"):                
                return True
            else:
                return False

        #if (obj1.type == "mushroom") and (obj1.color == "red" or obj1.color == "pink"):
        if (obj1.name == "mushroom"):
            print("HYPOTHESIS 1 CONDITION MET")
            #print("\t" + obj1.color)
            return self.testAssertion(exampleAssertion, [obj1])

        return False        


    def exampleHypothesis2(self, agent, obj1):
        # If an agent eats a mushroom, then it will become sick within the future. 
        def exampleAssertion(self, agent, obj1):
            # Check that 'agent' has 'isPoisoned' = true
            if (agent.isPoisoned == True):
                return True
            else:
                return False
                
        #if (obj1.name == "mushroom") and (agent.tookAction("EAT", obj1)):
        if (agent.tookAction("EAT", obj1)):
            #self.testHypothesisAssertion(agent.willBecomeSickWithin(50) == True)
            #return True
            print("HYPOTHESIS 2 CONDITION MET")
            return self.testAssertion(exampleAssertion, [agent, obj1])

        

        return False        


    # Assertions
    # Takes a function pointer to the assertion function, and the arguments to pass to the assertion function
    def testAssertion(self, assertionFunction, args):
        self.evaluationResults['conditionMetCount'] += 1

        #print("TESTING ASSERTION... (assertionFunction = " + str(assertionFunction) + ")")

        # Get the starting step
        startStep = self.curStep 

        # Get the ending step (the length of the world history)
        endStep = self.getWorldHistoryLength()

        # Iterate from the current step to the last step
        for step in range(startStep, endStep):
            # Get the world history at the current step
            allObjects, allObjectsByUUID = self.getWorldHistoryAtStep(step)
            # Get the objects with the same UUIDs as the arguments, but at the current time step
            argsAtStep = [self]
            for arg in args:
                if (arg.uuid in allObjectsByUUID):
                    argAtStep = allObjectsByUUID[arg.uuid]                
                    argsAtStep.append(argAtStep)
                else:
                    argsAtStep.append(None)
                    # TODO: Should we just stop here and return False, if the object has disappeared?  (Or do some other graceful error handling? But tricky -- e.g. the object might have been eaten, and isn't relevant any more)
            # Run the assertion function with the arguments
            assertionResult = False
            try:                
                #print("Running assertion function...")
                assertionResult = assertionFunction(*argsAtStep)
                #print("Assertion result = " + str(assertionResult) + " at step " + str(step) + " (of " + str(endStep) + ")")
            except KeyboardInterrupt:
                print("Keyboard interrupt")
                exit(1)
            except Exception as e:
                #print("Exception occurred: " + str(e))
                assertionResult = False

            # If the assertion is true, then return True
            if (assertionResult == True):
                self.evaluationResults['assertionMetCount'] += 1
                return True
            
            # Otherwise, continue looking for one or more cases where the assertion is true.
            
        # If we get here, then the assertion was never found to be true, so return False
        return False


#
#   Main
#
if __name__ == "__main__":
    # Parse command line arguments to get the export filename 
    # (e.g. python KnowledgeEvaluationTemplate.py --exportFilename=evaluationResults.json)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exportFilename", type=str, default="evaluationResults.json", help="Filename to export the evaluation results to")
    args = parser.parse_args()

    # Step 1: Initialize the knowledge evaluation template (and load the world history)
    knowledgeEvaluationTemplate = KnowledgeEvaluationTemplate()

    # Step 2: Run the hypothesis
    # TODO
    
    worldHistoryLength = knowledgeEvaluationTemplate.getWorldHistoryLength()
    print("World history has " + str(worldHistoryLength) + " steps.")

    for step in range(1, worldHistoryLength):
#        print("")
#        print("\n------------------\n")

        print("Checking step " + str(step) + ":")

        allObjects0, allObjectsByUUID0 = knowledgeEvaluationTemplate.getWorldHistoryAtStep(step)        # Current step
        allObjects1, allObjectsByUUID1 = knowledgeEvaluationTemplate.getWorldHistoryAtStep(step-1)      # Previous step

        allObjects = []
        allObjectsByUUID = {}
        
        # If the object doesn't have an action history, then use its version from the last step
        # NOTE: This is a hack, because while the agent's action history will present as "What it's about to do", the agent's properties will present as "What it used to be"
        for obj in allObjects1:
            # If there's no action history, then use the object from the previous step
            if (obj.actionHistory == None) or (len(obj.actionHistory) == 0):
                allObjects.append(obj)
                allObjectsByUUID[obj.uuid] = obj
            # Otherwise, use the object from the current step
            else:
                objAtCurrent = allObjectsByUUID0[obj.uuid]
                allObjects.append(objAtCurrent)
                allObjectsByUUID[obj.uuid] = objAtCurrent            
                
        #allObjects, allObjectsByUUID = knowledgeEvaluationTemplate.getWorldHistoryAtStep(5)
        
        # print("Evaluating Hypothesis 1...")
        # for obj in allObjects:
        #     result = knowledgeEvaluationTemplate.exampleHypothesis1(obj)
        #     if (result == True):
        #         print("\tHypothesis 1 is true for object: " + obj.name)
        #         #print(str(obj.attributes))
        

        print("Evaluating Hypothesis 2...")
        for obj1 in allObjects:
            for obj2 in allObjects:
                result = knowledgeEvaluationTemplate.exampleHypothesis2(obj1, obj2)
                if (result == True):
                    print("\tHypothesis 2 is true for objects: " + obj1.name + ", " + obj2.name)
                    #print(str(obj1.attributes))
                    #print(str(obj2.attributes))


        print("Checking for object with specific UUID...")
        uuidToCheck = 830024008     # Mushroom that is eaten in step 14
        for obj in allObjects:
            if (obj.uuid == uuidToCheck):
                print("\tFound object with UUID " + str(uuidToCheck) + ": " + obj.name)
                #print(str(obj.attributes))
#        print("")
#        print("\n------------------\n")

        print("Evaluation results: " + str(knowledgeEvaluationTemplate.evaluationResults))


    # Step 3: Export the results (use command line argument to specify filename)
    knowledgeEvaluationTemplate.exportEvaluationResults(args.exportFilename)
    # Print the evaluation results to the console
    print("Evaluation results:")
    print(knowledgeEvaluationTemplate.evaluationResults)