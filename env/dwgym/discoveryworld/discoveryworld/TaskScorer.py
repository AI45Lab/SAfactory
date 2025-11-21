# TaskScorer.py
from discoveryworld.Agent import NPC
from discoveryworld.objects import *
from discoveryworld.ActionHistory import *


#
#   Task Scorer
#   Holds all active tasks for a given world
#
class TaskScorer():
    # Constructor
    def __init__(self, world):
        self.tasks = []
        self.world = world


    # Add a task to the list
    def addTask(self, task):
        self.tasks.append(task)


    # Update the task progress
    def updateTick(self):
        for task in self.tasks:
            task.updateTick()


    # String representing task progress
    def taskProgressStr(self):
        outStr = "Task Progress:\n"
        for idx, task in enumerate(self.tasks):
            outStr += str(idx) + ": " + task.taskProgressStr() + "\n"

        return outStr



#
#   Task Maker
#   A generator for going from task name to a specific Task object
#
class TaskMaker():
    # Constructor
    def __init__(self, world):
        self.world = world

    # Make a task
    def makeTask(self, taskName:str, scoringInfo:dict=None):
        # Discovery tasks
        if (taskName == "SpaceSickTaskNormal"):
            return SpaceSickTaskNormal(self.world, scoringInfo)
        elif (taskName == "SpaceSickTaskEasy"):
            from discoveryworld.scenarios.space_sick_easy import SpaceSickTaskEasy
            return SpaceSickTaskEasy(self.world, scoringInfo)
        elif (taskName == "SpaceSickTaskChallenge"):
            return SpaceSickTaskChallenge(self.world, scoringInfo)
        elif (taskName == "RustedKeyTaskEasy"):
            return RustedKeyTaskEasy(self.world, scoringInfo)
        elif (taskName == "RustedKeyTaskNormal"):
            return RustedKeyTaskNormal(self.world, scoringInfo)
        elif (taskName == "RustedKeyTaskChallenge"):
            return RustedKeyTaskChallenge(self.world, scoringInfo)
        elif (taskName == "ArchaeologyDigTaskEasy"):
            return ArchaeologyDigEasy(self.world, scoringInfo)
        elif (taskName == "ArchaeologyDigTaskNormal"):
            return ArchaeologyDigNormal(self.world, scoringInfo)
        elif (taskName == "ArchaeologyDigTaskGenericRadioisotope"):
            return ArchaeologyDigGenericRadioisotopes(self.world, scoringInfo)
        elif (taskName == "SoilNutrientTaskEasy"):
            return SoilNutrientTaskEasy(self.world, scoringInfo)
        elif (taskName == "SoilNutrientTaskNormal"):
            return SoilNutrientTaskNormal(self.world, scoringInfo)
        elif (taskName == "SoilNutrientTaskChallenge"):
            return SoilNutrientTaskChallenge(self.world, scoringInfo)
        elif (taskName == "RosettaStoneTaskEasy"):
            from discoveryworld.scenarios.rosetta_stone_easy import RosettaStoneTaskEasy
            return RosettaStoneTaskEasy(self.world, scoringInfo)
        elif (taskName == "RosettaStoneTask"):
            return RosettaStoneTask(self.world, scoringInfo)
        elif (taskName == "TutorialTask"):
            return TutorialTask(self.world, scoringInfo)
        elif (taskName == "ReactorTaskEasy"):
            return ReactorTaskEasy(self.world, scoringInfo)
        elif (taskName == "ReactorTaskNormal"):
            return ReactorTaskNormal(self.world, scoringInfo)
        elif (taskName == "ReactorTaskChallenge"):
            return ReactorTaskChallenge(self.world, scoringInfo)
        elif (taskName == "ProteomicsTaskEasy"):
            return ProteomicsTaskEasy(self.world, scoringInfo)
        elif (taskName == "ProteomicsTaskNormal"):
            return ProteomicsTask(self.world, scoringInfo, challengeVersion=False)
        elif (taskName == "ProteomicsTaskChallenge"):
            return ProteomicsTask(self.world, scoringInfo, challengeVersion=True)
        elif (taskName == "NotRocketScienceTask"):
            from discoveryworld.scenarios.not_rocket_science import NotRocketScienceTask
            return NotRocketScienceTask(self.world, scoringInfo)

        # Small Skills
        elif (taskName == "SmallSkillsDialogTask"):
            from discoveryworld.scenarios import SmallSkillsDialogTask
            return SmallSkillsDialogTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsPickAndPlaceTask"):
            from discoveryworld.scenarios import SmallSkillsPickAndPlaceTask
            return SmallSkillsPickAndPlaceTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsPickAndGiveTask"):
            from discoveryworld.scenarios import SmallSkillsPickAndGiveTask
            return SmallSkillsPickAndGiveTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsInstrumentMeasurementTask"):
            from discoveryworld.scenarios import SmallSkillsInstrumentMeasurementTask
            return SmallSkillsInstrumentMeasurementTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsDoorsTask"):
            from discoveryworld.scenarios import SmallSkillsDoorsTask
            return SmallSkillsDoorsTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsDoorsKeysTask"):
            from discoveryworld.scenarios import SmallSkillsDoorsKeysTask
            return SmallSkillsDoorsKeysTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsNavigationHouseTask"):
            from discoveryworld.scenarios import SmallSkillsNavigationHouseTask
            return SmallSkillsNavigationHouseTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsSearchTask"):
            from discoveryworld.scenarios import SmallSkillsSearchTask
            return SmallSkillsSearchTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsDiscoveryFeedTask"):
            from discoveryworld.scenarios import SmallSkillsDiscoveryFeedTask
            return SmallSkillsDiscoveryFeedTask(self.world, scoringInfo)
        elif (taskName == "SmallSkillsMovingAgentsTask"):
            from discoveryworld.scenarios import SmallSkillsMovingAgentsTask
            return SmallSkillsMovingAgentsTask(self.world, scoringInfo)

        else:
            print("ERROR: UNKNOWN TASK NAME: " + taskName)
            exit(1)
            return None

#
#   Abstract class template for a specific task
#
class Task():
    # Constructor
    def __init__(self, taskName:str, taskDescription:str, world, scoringInfo:dict):
        self.taskName = taskName
        self.taskDescription = taskDescription
        self.world = world
        self.scoringInfo = scoringInfo
        self.score = 0                          # Current task score
        self.maxScore = 1                       # Maximum score
        self.completed = False                  # Is this task over?
        self.completedSuccessfully = False      # Did the task complete successfully?
        self.scoreCard = []                     # List of the subtasks in this task's scorecard
        self.criticalHypotheses = []            # List of critical hypotheses for this task
        self.criticalQuestions = []


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        pass

    def getScore(self):
        return self.score

    def getScoreNormalized(self):
        return self.score / self.maxScore

    def isCompleted(self):
        return self.completed

    def isCompletedSuccessfully(self):
        return self.completedSuccessfully

    # Update the task progress
    def updateTick(self):
        pass

    # String representing task progress
    def taskProgressStr(self):
        outStr = ""
        outStr += "Task Progress (" + self.taskName + "):\n"
        outStr += "\tTask Description: " + self.taskDescription + "\n"
        outStr += "\tScore: " + str(self.score) + ", MaxScore: " + str(self.maxScore) + ", Normalized Score: " + str(format(self.getScoreNormalized(), '.2f')) + ", Completed: " + str(self.completed) + ", Completed Successfully: " + str(self.completedSuccessfully) + "\n"
        outStr += "Scorecard:"
        for element in self.scoreCard:
            #outStr += "\n" + element.name + ": " + str(element.score) + "/" + str(element.maxScore) + ", Completed: " + str(element.completed)
            # As above, but with constant witdh formatting, completed should be 'completed' or 'not completed', and the task description should be there too
            isCompleted = "incomplete"
            if (element.completed == True):
                isCompleted = "completed"
            outStr += "\n - " + element.name.ljust(40) + " " + str(element.score).rjust(2) + " /" + str(element.maxScore).rjust(2) + "  " + str(isCompleted).ljust(15) + " " + element.description
        # Critical Hypotheses
        outStr += "\nCritical Hypotheses:"
        for hypothesis in self.criticalHypotheses:
            outStr += "\n - " + hypothesis
        outStr += "\nCritical Questions:"
        for question in self.criticalQuestions:
            outStr += "\n - " + question


        return outStr

    # Get task progress as a dictionary
    def taskProgressDict(self):
        outDict = {}

        outDict["taskName"] = self.taskName
        outDict["taskDescription"] = self.taskDescription
        outDict["score"] = self.score
        outDict["maxScore"] = self.maxScore
        outDict["scoreNormalized"] = self.getScoreNormalized()
        outDict["completed"] = self.completed
        outDict["completedSuccessfully"] = self.completedSuccessfully
        outDict["scoreCard"] = self.getScorecard()
        outDict["criticalHypotheses"] = self.criticalHypotheses
        outDict["criticalQuestions"] = self.criticalQuestions

        return outDict

    # Get the scorecard elements as a form that can be serialized to JSON
    def getScorecard(self):
        out = []
        for element in self.scoreCard:
            out.append(element.toDict())
        return out




#
#   Scorecard element: A specific subtask of a larger task
#
class ScorecardElement():
    # Constructor
    def __init__(self, name:str, description:str, maxScore:int):
        self.name = name
        self.description = description
        self.maxScore = maxScore
        self.score = 0
        self.completed = False
        self.associatedUUIDs = []       # List of UUIDs of objects associated with this scorecard element
        self.associatedNotes = ""       # Notes associated with this scorecard element

    # Update the score
    def updateScore(self, score:int, completed:bool, associatedUUIDs:list=[], associatedNotes:str=""):
        assert score <= self.maxScore, f"Score exceeds maximum score for this element: {score} > {self.maxScore}"
        self.score = score
        self.completed = completed
        self.associatedUUIDs = associatedUUIDs
        self.associatedNotes = associatedNotes

    def toDict(self):
        out = {}
        out["name"] = self.name
        out["description"] = self.description
        out["maxScore"] = self.maxScore
        out["score"] = self.score
        out["completed"] = self.completed
        out["associatedUUIDs"] = self.associatedUUIDs
        out["associatedNotes"] = self.associatedNotes
        return out


#
#   Specific Task: Agents eating space mushrooms without getting sick
#
class SpaceSickTaskNormal(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "The only food on this planet are local mushrooms, but after eating them, the colonist sometimes have upset stomachs.  Your task is to figure out why people are feeling ill, and to prevent it.  You must demonstrate this by having colonists successfully eat 10 mushrooms in a row without eventually feeling sick. "
        taskDescription += "Since the food causes only mild illness, and getting the colony established is important, the colonists have volunteered to be test subjects.  The Chef in the Cafeteria can help you collect mushrooms, serve mushrooms from the cafeteria pot to the tables, and let the colonists know a meal is ready to eat, when you're ready. "
        taskDescription += "The colonists may post their status (like if they're feeling unwell) on the Discovery Feed (use 'v' to display it)."
        taskDescription += "After a colonist eats a mushroom, it will be automatically monitored by DiscoveryWorld for 50 turns to see if it gets sick.  (Note: If it successfully eats another mushroom within that 50 turns, assuming both are good, it will still only count as a single good case.)"
        Task.__init__(self, "SpaceSickTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 10                       # TODO: Maximum score
        self.agentsToMonitorForSickness = {}        # Key: agent name, value: step they were added

        # TODO: Add subtasks?
    # instruments['microscope'] = world.createObject("Microscope")
    # instruments['spectrometer'] = world.createObject("Spectrometer")
    # instruments['phmeter'] = world.createObject("PHMeter")
    # instruments['radiationmeter'] = world.createObject("RadiationMeter")
    # instruments['sampler'] = world.createObject("Sampler")
    # instruments['thermometer'] = world.createObject("Thermometer")
    # instruments['npkmeter'] = world.createObject("NPKMeter")

        # Have collected at least one of each of the 4 different colors of mushrooms
        self.collectedMushroomColors = set()
        self.collectedMushroomUUIDs = set()
        self.scorecardMushrooms = ScorecardElement("Collect different mushrooms", "Collect at least one of each of the 4 different colors of mushrooms", maxScore=4)
        self.scoreCard.append(self.scorecardMushrooms)

        # Use all the different scientific instruments on something
        self.scorecardInstruments = ScorecardElement("Use instruments", "Use each of the scientific instruments on an object", maxScore=7)
        self.scoreCard.append(self.scorecardInstruments)
        self.scorecardInstruments2 = ScorecardElement("Use instruments on mushrooms", "Use each of the scientific instruments on a mushroom", maxScore=7)
        self.scoreCard.append(self.scorecardInstruments2)

        # Have at least 10 mushrooms eaten by colonists
        self.scorecardMushroomsEaten = ScorecardElement("Eat mushrooms", "Have at least 10 mushrooms eaten by colonists", maxScore=10)
        self.scoreCard.append(self.scorecardMushroomsEaten)

        # Have 10 mushrooms eaten by colonists without them having got sick
        self.scorecardMushroomsEatenNoSickness = ScorecardElement("Eat mushrooms without sickness", "Have 10 mushrooms eaten by colonists without them getting sick", maxScore=10)
        self.scoreCard.append(self.scorecardMushroomsEatenNoSickness)

        # Store the colonists
        self.colonists = scoringInfo['colonists']
        self.numAgentsSuccessfullyEatenMushrooms = 0

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    # Update the task progress
    def updateTick(self):
        # Score Element 1: Collect different mushrooms
        # Check what mushrooms the agent has in their inventory
        if (not self.scorecardMushrooms.completed):
            for agent in self.world.getUserAgents():
                for obj in agent.getAllContainedObjectsAndParts():
                    if ("mushroom" in obj.type):
                        mushroomColor = obj.attributes['color']
                        if (mushroomColor not in self.collectedMushroomColors):
                            self.collectedMushroomColors.add(obj.attributes['color'])
                            self.collectedMushroomUUIDs.add(obj.uuid)
            completedColors = False
            if (len(self.collectedMushroomColors) >= 4):
                completedColors = True
            self.scorecardMushrooms.updateScore(len(self.collectedMushroomColors), completedColors, associatedUUIDs=list(self.collectedMushroomUUIDs), associatedNotes="The following mushroom colors have been collected: " + str(self.collectedMushroomColors))

        # Score Element 2: Use different scientific instruments on a mushroom
        # Check if the agent has used each of the scientific instruments on a mushroom
        if (not self.scorecardInstruments.completed) or (not self.scorecardInstruments2.completed):
            usedInstrumentsUUIDs = set()
            usedInstrumentNames = set()
            usedInstrumentsUUIDs2 = set()
            usedInstrumentNames2 = set()
            for agent in self.world.getUserAgents():
                for instrument in self.scoringInfo['instruments'].values():
                    # Check if the agent has used the instrument on anything
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2="*", stopAtFirst=False)
                    if (len(foundActions) > 0):
                        usedInstrumentsUUIDs.add(instrument.uuid)
                        usedInstrumentNames.add(instrument.name)

                    # Check if the agent has used the instrument on a mushroom
                    #def queryActionObjectsByArgType(self, actionType:ActionType, arg1=None, arg2TypeContains="", stopAtFirst:bool = False):
                    foundActionsMushroom = agent.actionHistory.queryActionObjectsByArg2Type(ActionType.USE, arg1=instrument, arg2TypeContains="mushroom", stopAtFirst=True)
                    if (len(foundActionsMushroom) > 0):
                        usedInstrumentsUUIDs2.add(instrument.uuid)
                        usedInstrumentNames2.add(instrument.name)

            completed1 = False
            if (len(usedInstrumentNames) == 7):
                completed1 = True
            self.scorecardInstruments.updateScore(len(usedInstrumentNames), completed1, associatedUUIDs=list(usedInstrumentsUUIDs), associatedNotes="The following instruments have been used: " + str(usedInstrumentNames))
            completed2 = False
            if (len(usedInstrumentNames2) == 7):
                completed2 = True
            self.scorecardInstruments2.updateScore(len(usedInstrumentNames2), completed2, associatedUUIDs=list(usedInstrumentsUUIDs2), associatedNotes="The following instruments have been used on a mushroom: " + str(usedInstrumentNames))


        # Score Element 3: Have at least 10 mushrooms eaten by colonists
        if (not self.scorecardMushroomsEaten.completed):
            associatedUUIDs = set()
            for colonist in self.colonists:
                foundActions = colonist.actionHistory.queryActionObjectsByArg1Type(ActionType.EAT, arg1TypeContains="mushroom", stopAtFirst=False)
                for action in foundActions:
                    associatedUUIDs.add(action['arg1'].uuid)

            numEaten = len(associatedUUIDs)

            completedEaten = False
            if (numEaten >= 10):
                completedEaten = True
            if (completedEaten == True):
                self.scorecardMushroomsEaten.updateScore(numEaten, completedEaten, associatedUUIDs=list(associatedUUIDs), associatedNotes="At least 10 mushrooms have been eaten by colonists")
            else:
                self.scorecardMushroomsEaten.updateScore(numEaten, completedEaten, associatedUUIDs=list(associatedUUIDs), associatedNotes= str(numEaten) + " mushrooms have been eaten by colonists")





        # Monitoring task 1: Check if any agents have just eaten a mushroom
        # List of names of agents to check for whether they've just eaten a mushroom
        ## NOTE: Slightly hacky because this was early code
        agentsToCheck = []
        for i in range (0, 5):
            agentsToCheck.append("Colonist " + str(i))

        for agent in self.world.agents:
            if agent.name in agentsToCheck:
                # Check if they have eaten a mushroom

                # Check if they successfully ate something on the last tick
                lastAction = agent.actionHistory.getLastStepAction()
                if (lastAction != None) and (lastAction['actionType'] == ActionType.EAT) and (lastAction['success'] == True):
                    # Check if it was a mushroom
                    if (lastAction['arg1'].name == "mushroom"):
                        #self.score += 1
                        self.agentsToMonitorForSickness[agent.name] = self.world.getStepCounter()


        # Monitoring task 2: Check if any agents that recently ate a mushroom are now sick
        agentsToCheckForSickess = list(self.agentsToMonitorForSickness.keys())      # Deep copy, so we can delete from the original without changing keys while iterating
        for agentName in agentsToCheckForSickess:
            # Check if the agent is sick
            agent = self.world.getAgentByName(agentName)
            if (agent.attributes['isPoisoned'] == True):
                # Reset score to 0
                #print("Agent " + agentName + " is sick!  Score reset to 0.")
                self.numAgentsSuccessfullyEatenMushrooms = 0
            else:
                # Check if the agent has been well for 50 steps
                if (self.world.getStepCounter() - self.agentsToMonitorForSickness[agentName] >= 50):
                    # If they've been well for 50 steps, increment the score, and remove the monitor
                    self.numAgentsSuccessfullyEatenMushrooms += 1
                    del self.agentsToMonitorForSickness[agentName]

        completedEaten = False
        if (self.numAgentsSuccessfullyEatenMushrooms >= 10):
            completedEaten = True
        self.scorecardMushroomsEatenNoSickness.updateScore(self.numAgentsSuccessfullyEatenMushrooms, completedEaten, associatedUUIDs=[], associatedNotes= str(self.numAgentsSuccessfullyEatenMushrooms) + " mushrooms have been eaten by colonists without them getting sick")

        # Update score
        score = 0
        maxScore = 0
        for element in self.scoreCard:
            score += element.score
            maxScore += element.maxScore
        self.score = score
        self.maxScore = maxScore

        # Monitoring task 3: Check if the task is complete
        if (completedEaten):
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)
        else:
            self.completed = False
            self.completedSuccessfully = False


#
#   Specific Task: Agents eating space mushrooms without getting sick
#
class SpaceSickTaskChallenge(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "The only food on this planet are local mushrooms, but after eating them, the colonist sometimes have upset stomachs.  Your task is to figure out why people are feeling ill, and to prevent it.  You must demonstrate this by having colonists successfully eat 10 mushrooms in a row without eventually feeling sick. "
        taskDescription += "Since the food causes only mild illness, and getting the colony established is important, the colonists have volunteered to be test subjects.  The Chef in the Cafeteria can help you collect mushrooms, serve mushrooms from the cafeteria pot to the tables, and let the colonists know a meal is ready to eat, when you're ready. "
        taskDescription += "The colonists may post their status (like if they're feeling unwell) on the Discovery Feed (use 'v' to display it)."
        taskDescription += "After a colonist eats a mushroom, it will be automatically monitored by DiscoveryWorld for 50 turns to see if it gets sick.  (Note: If it successfully eats another mushroom within that 50 turns, assuming both are good, it will still only count as a single good case.)"
        Task.__init__(self, "SpaceSickTaskChallenge", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 10                       # TODO: Maximum score
        self.agentsToMonitorForSickness = {}        # Key: agent name, value: step they were added

        # TODO: Add subtasks?
    # instruments['microscope'] = world.createObject("Microscope")
    # instruments['spectrometer'] = world.createObject("Spectrometer")
    # instruments['phmeter'] = world.createObject("PHMeter")
    # instruments['radiationmeter'] = world.createObject("RadiationMeter")
    # instruments['sampler'] = world.createObject("Sampler")
    # instruments['thermometer'] = world.createObject("Thermometer")
    # instruments['npkmeter'] = world.createObject("NPKMeter")

        # Have collected at least one of each of the 4 different colors of mushrooms
        self.collectedMushroomColors = set()
        self.collectedMushroomUUIDs = set()
        self.scorecardMushrooms = ScorecardElement("Collect different mushrooms", "Collect at least one of each of the 4 different colors of mushrooms", maxScore=4)
        self.scoreCard.append(self.scorecardMushrooms)

        # Use all the different scientific instruments on something
        self.scorecardInstruments = ScorecardElement("Use instruments", "Use each of the scientific instruments on an object", maxScore=7)
        self.scoreCard.append(self.scorecardInstruments)
        self.scorecardInstruments2 = ScorecardElement("Use instruments on mushrooms", "Use each of the scientific instruments on a mushroom", maxScore=7)
        self.scoreCard.append(self.scorecardInstruments2)

        # Have at least one of the glowing rocks in the agents inventory
        self.scorecardGlowingRock = ScorecardElement("Collect glowing rock", "Collect at least one of the glowing rocks", maxScore=1)
        self.scoreCard.append(self.scorecardGlowingRock)

        # Have a glowing rock turn luminous
        self.scorecardGlowingRockLuminous = ScorecardElement("Glowing rock luminous", "Have a glowing rock turn luminous", maxScore=1)
        self.scoreCard.append(self.scorecardGlowingRockLuminous)

        # Have at least 10 mushrooms eaten by colonists
        self.scorecardMushroomsEaten = ScorecardElement("Eat mushrooms", "Have at least 10 mushrooms eaten by colonists", maxScore=10)
        self.scoreCard.append(self.scorecardMushroomsEaten)

        # Have 10 mushrooms eaten by colonists without them having got sick
        self.scorecardMushroomsEatenNoSickness = ScorecardElement("Eat mushrooms without sickness", "Have 10 mushrooms eaten by colonists without them getting sick", maxScore=10)
        self.scoreCard.append(self.scorecardMushroomsEatenNoSickness)

        # Store the colonists
        self.colonists = scoringInfo['colonists']
        self.numAgentsSuccessfullyEatenMushrooms = 0

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    # Update the task progress
    def updateTick(self):
        # Score Element 1: Collect different mushrooms
        # Check what mushrooms the agent has in their inventory
        if (not self.scorecardMushrooms.completed):
            for agent in self.world.getUserAgents():
                for obj in agent.getAllContainedObjectsAndParts():
                    if ("mushroom" in obj.type):
                        mushroomColor = obj.attributes['color']
                        if (mushroomColor not in self.collectedMushroomColors):
                            self.collectedMushroomColors.add(obj.attributes['color'])
                            self.collectedMushroomUUIDs.add(obj.uuid)
            completedColors = False
            if (len(self.collectedMushroomColors) >= 4):
                completedColors = True
            self.scorecardMushrooms.updateScore(len(self.collectedMushroomColors), completedColors, associatedUUIDs=list(self.collectedMushroomUUIDs), associatedNotes="The following mushroom colors have been collected: " + str(self.collectedMushroomColors))

        # Have a glowing rock in the agents inventory
        if (not self.scorecardGlowingRock.completed):
            glowingRockFound = False
            glowingRockUUIDs = set()
            for agent in self.world.getUserAgents():
                for obj in agent.getAllContainedObjectsAndParts():
                    if ("rock (glowing)" in obj.type):
                        glowingRockFound = True
                        glowingRockUUIDs.add(obj.uuid)
            if (glowingRockFound == True):
                self.scorecardGlowingRock.updateScore(1, glowingRockFound, associatedUUIDs=list(glowingRockUUIDs), associatedNotes="A glowing rock has been collected")

        # Have any glowing rock turn luminous
        if (not self.scorecardGlowingRockLuminous.completed):
            for obj in self.scoringInfo["glowingRocks"]:
                if ("isLuminous" in obj.attributes) and (obj.attributes['isLuminous'] == True):
                    self.scorecardGlowingRockLuminous.updateScore(1, True, associatedUUIDs=[obj.uuid], associatedNotes="A glowing rock has turned luminous")

        # Score Element 2: Use different scientific instruments on a mushroom
        # Check if the agent has used each of the scientific instruments on a mushroom
        if (not self.scorecardInstruments.completed) or (not self.scorecardInstruments2.completed):
            usedInstrumentsUUIDs = set()
            usedInstrumentNames = set()
            usedInstrumentsUUIDs2 = set()
            usedInstrumentNames2 = set()
            for agent in self.world.getUserAgents():
                for instrument in self.scoringInfo['instruments'].values():
                    # Check if the agent has used the instrument on anything
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2="*", stopAtFirst=False)
                    if (len(foundActions) > 0):
                        usedInstrumentsUUIDs.add(instrument.uuid)
                        usedInstrumentNames.add(instrument.name)

                    # Check if the agent has used the instrument on a mushroom
                    #def queryActionObjectsByArgType(self, actionType:ActionType, arg1=None, arg2TypeContains="", stopAtFirst:bool = False):
                    foundActionsMushroom = agent.actionHistory.queryActionObjectsByArg2Type(ActionType.USE, arg1=instrument, arg2TypeContains="mushroom", stopAtFirst=True)
                    if (len(foundActionsMushroom) > 0):
                        usedInstrumentsUUIDs2.add(instrument.uuid)
                        usedInstrumentNames2.add(instrument.name)

            completed1 = False
            if (len(usedInstrumentNames) == 7):
                completed1 = True
            self.scorecardInstruments.updateScore(len(usedInstrumentNames), completed1, associatedUUIDs=list(usedInstrumentsUUIDs), associatedNotes="The following instruments have been used: " + str(usedInstrumentNames))
            completed2 = False
            if (len(usedInstrumentNames2) == 7):
                completed2 = True
            self.scorecardInstruments2.updateScore(len(usedInstrumentNames2), completed2, associatedUUIDs=list(usedInstrumentsUUIDs2), associatedNotes="The following instruments have been used on a mushroom: " + str(usedInstrumentNames))


        # Score Element 3: Have at least 10 mushrooms eaten by colonists
        if (not self.scorecardMushroomsEaten.completed):
            associatedUUIDs = set()
            for colonist in self.colonists:
                foundActions = colonist.actionHistory.queryActionObjectsByArg1Type(ActionType.EAT, arg1TypeContains="mushroom", stopAtFirst=False)
                for action in foundActions:
                    associatedUUIDs.add(action['arg1'].uuid)

            numEaten = len(associatedUUIDs)

            completedEaten = False
            if (numEaten >= 10):
                completedEaten = True
            if (completedEaten == True):
                self.scorecardMushroomsEaten.updateScore(numEaten, completedEaten, associatedUUIDs=list(associatedUUIDs), associatedNotes="At least 10 mushrooms have been eaten by colonists")
            else:
                self.scorecardMushroomsEaten.updateScore(numEaten, completedEaten, associatedUUIDs=list(associatedUUIDs), associatedNotes= str(numEaten) + " mushrooms have been eaten by colonists")


        # Monitoring task 1: Check if any agents have just eaten a mushroom
        # List of names of agents to check for whether they've just eaten a mushroom
        ## NOTE: Slightly hacky because this was early code
        agentsToCheck = []
        for i in range (0, 5):
            agentsToCheck.append("Colonist " + str(i))

        for agent in self.world.agents:
            if agent.name in agentsToCheck:
                # Check if they have eaten a mushroom

                # Check if they successfully ate something on the last tick
                lastAction = agent.actionHistory.getLastStepAction()
                if (lastAction != None) and (lastAction['actionType'] == ActionType.EAT) and (lastAction['success'] == True):
                    # Check if it was a mushroom
                    if (lastAction['arg1'].name == "mushroom"):
                        #self.score += 1
                        self.agentsToMonitorForSickness[agent.name] = self.world.getStepCounter()


        # Monitoring task 2: Check if any agents that recently ate a mushroom are now sick
        agentsToCheckForSickess = list(self.agentsToMonitorForSickness.keys())      # Deep copy, so we can delete from the original without changing keys while iterating
        for agentName in agentsToCheckForSickess:
            # Check if the agent is sick
            agent = self.world.getAgentByName(agentName)
            if (agent.attributes['isPoisoned'] == True):
                # Reset score to 0
                #print("Agent " + agentName + " is sick!  Score reset to 0.")
                self.numAgentsSuccessfullyEatenMushrooms = 0
            else:
                # Check if the agent has been well for 50 steps
                if (self.world.getStepCounter() - self.agentsToMonitorForSickness[agentName] >= 50):
                    # If they've been well for 50 steps, increment the score, and remove the monitor
                    self.numAgentsSuccessfullyEatenMushrooms += 1
                    del self.agentsToMonitorForSickness[agentName]

        completedEaten = False
        if (self.numAgentsSuccessfullyEatenMushrooms >= 10):
            completedEaten = True
        self.scorecardMushroomsEatenNoSickness.updateScore(self.numAgentsSuccessfullyEatenMushrooms, completedEaten, associatedUUIDs=[], associatedNotes= str(self.numAgentsSuccessfullyEatenMushrooms) + " mushrooms have been eaten by colonists without them getting sick")

        # Update score
        score = 0
        maxScore = 0
        for element in self.scoreCard:
            score += element.score
            maxScore += element.maxScore
        self.score = score
        self.maxScore = maxScore

        # Monitoring task 3: Check if the task is complete
        if (completedEaten):
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)
        else:
            self.completed = False
            self.completedSuccessfully = False



#
#   Specific Task: Rusted key task
#
class RustedKeyTaskNormal(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "You were venturing into the wilderness of Planet X to a storage shed to fetch some supplies. The shed door locked behind you, and your key appears too rusted to work in the lock. "
        taskDescription += "You remember one of the other colonists mentioning that some combination of the chemicals in the shed could make a rust remover, but you can't remember the details. "
        taskDescription += "You need to figure out a way to remove the rust from the key, and use it to open the shed door, and make your way back to the colony. "
        taskDescription += "You have no means of communication with the other colonists, and must figure this out on your own. \n"
        taskDescription += "Some helpful notes: \n"
        taskDescription += "1. The rusted key is in the shed. \n"
        taskDescription += "2. The shed door is locked, and will only open using a non-rusted key. \n"
        taskDescription += "3. You can mix chemicals in the jar. Just use a chemical dispenser on the jar to add chemicals to the jar. \n"
        taskDescription += "4. Chemicals placed in the same jar will automatically mix, though it may take up to 2 steps for this to happen. \n"
        taskDescription += "5. Placing the rusted key into the jar will automatically apply the chemicals to the key.  If the chemical mixture is correct, the rust will be removed, though it may up to 2 steps for this to happen. \n"
        taskDescription += "6. If you need to clean the jar of chemicals to try a new combination, use the bottle washer on the jar. \n"
        taskDescription += "7. When you have successfully derusted the key and opened the door, please leave the shed. \n"
        ## taskDescription += "HERE'S A WALKTHROUGH/HINT: To complete this task, you need to pick up the jar, put 1 unit of Chemical A and 2 units of Chemical C into the jar, and then put the key into the jar.  The key will change from rusted to not rusted.  Then you can open the door, walk 3 steps out, and the task will be completed."

        Task.__init__(self, "RustedKeyTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 2                       # Maximum score

        #scoringInfo['dispensers'] = [dispenser1, dispenser2, dispenser3]
        #scoringInfo['mixingJar'] = mixingJar
        #scoringInfo['bottleCleaner'] = BottleCleaner
        #scoringInfo['key'] = rustyKey

        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardKey = ScorecardElement("Take key", "The key has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardKey)
        self.scorecardJar = ScorecardElement("Take jar", "The mixing jar has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardJar)

        # Has used each chemical dispenser
        self.scorecardUsedDispensers = ScorecardElement("Use each chemical dispenser", "Each chemical dispenser has been used", maxScore=3)
        self.scoreCard.append(self.scorecardUsedDispensers)

        # Has used the bottle cleaner
        self.scorecardUsedBottleCleaner = ScorecardElement("Use bottle cleaner", "The bottle cleaner has been used", maxScore=1)
        self.scoreCard.append(self.scorecardUsedBottleCleaner)

        # Has a mixture containing more than one chemical in the jar
        self.scorecardMixtureInJar = ScorecardElement("Mix chemicals in jar", "The jar contains a mixture of chemicals", maxScore=1)
        self.scoreCard.append(self.scorecardMixtureInJar)

        # Has placed the key in the jar
        self.scorecardKeyInJar = ScorecardElement("Place key in jar", "The key has been placed in the jar at least once", maxScore=1)
        self.scoreCard.append(self.scorecardKeyInJar)

        # Key is not rusted
        self.scorecardKeyNotRusted = ScorecardElement("Key is not rusted", "The key is not rusted (includes levels for reduced amounts of rust)", maxScore=3)
        self.scoreCard.append(self.scorecardKeyNotRusted)

        # Agent is outside
        self.scorecardAgentOutside = ScorecardElement("Agent is outside", "The agent is outside the shed", maxScore=1)
        self.scoreCard.append(self.scorecardAgentOutside)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check whether key has been in agents inventory
        if (not self.scorecardKey.completed):
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.type == "agent"):
                self.scorecardKey.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been in an agent's (UUID: " + str(key.parentContainer.uuid) + ") inventory")

        # Check whether jar has been in agents inventory
        if (not self.scorecardJar.completed):
            jar = self.scoringInfo['mixingJar']
            if (jar.parentContainer != None) and (jar.parentContainer.type == "agent"):
                self.scorecardJar.updateScore(1, True, associatedUUIDs=[jar.uuid, jar.parentContainer.uuid], associatedNotes="The jar has been in an agent's (UUID: " + str(jar.parentContainer.uuid) + ") inventory")

        # Check whether each chemical dispenser has been used
        if (not self.scorecardUsedDispensers.completed):
            usedDispensers = set()
            for agent in self.world.agents:
                for dispenser in self.scoringInfo['dispensers']:
                    # query the action history
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=dispenser, arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                    if (len(foundActions) > 0):
                        usedDispensers.add(dispenser.uuid)

            isCompleted = False
            if (len(usedDispensers) == 3):
                isCompleted = True
            associatedNotes = str(len(usedDispensers)) + " of " + str(len(self.scoringInfo["dispensers"])) + " dispensers have been used"
            self.scorecardUsedDispensers.updateScore(len(usedDispensers), isCompleted, associatedUUIDs=list(usedDispensers), associatedNotes=associatedNotes)

        # Check whether the bottle cleaner has been used
        if (not self.scorecardUsedBottleCleaner.completed):
            usedBottleCleaner = False
            for agent in self.world.agents:
                foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo['bottleCleaner'], arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                if (len(foundActions) > 0):
                    usedBottleCleaner = True
                    break
            if (usedBottleCleaner == True):
                self.scorecardUsedBottleCleaner.updateScore(1, usedBottleCleaner, associatedUUIDs=[self.scoringInfo['bottleCleaner'].uuid], associatedNotes="The bottle cleaner has been used")

        # Check whether the jar contains a mixture of chemicals
        if (not self.scorecardMixtureInJar.completed):
            # Check to see whether the jar contains a substance, and if the substance has a mixture (mixtureDict)
            for cObj in self.scoringInfo['mixingJar'].contents:
                if (cObj.type == "substance"):
                    mixtureSubstances = cObj.attributes['mixtureDict'].keys()
                    if (len(mixtureSubstances) > 1):
                        self.scorecardMixtureInJar.updateScore(1, True, associatedUUIDs=[cObj.uuid], associatedNotes="The mixing jar has contained two or more chemicals in a mixture (" + str(", ".join(mixtureSubstances)) + ")")
                        break

        # Check whether the key is in the jar
        if (not self.scorecardKeyInJar.completed):
            # Check whether the key's parent container is the jar
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.uuid == self.scoringInfo['mixingJar'].uuid):
                self.scorecardKeyInJar.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been placed in the jar at least once")

        # Check whether the key is not rusted
        if (not self.scorecardKeyNotRusted.completed):
            # Check the rust level of the key
            key = self.scoringInfo['key']
#        self.attributes['isRusted'] = isRusted                    # Is the key rusted?
#        self.attributes['rustLevel'] = 3 if isRusted else 0       # Description of the rust (0=none, 1=light, 2=medium, 3=heavy)
            if (key.attributes['isRusted'] == False):
                self.scorecardKeyNotRusted.updateScore(3, True, associatedUUIDs=[key.uuid], associatedNotes="The key is not rusted")
            else:
                # Partial
                rustScore = 3 - key.attributes['rustLevel']
                if (rustScore == 3):
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is fully rusted")
                else:
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is partially rusted")

        # Check to see whether one or more agents are outside of the room
        if (not self.scorecardAgentOutside.completed):
            # Bounds
            x0 = 16
            y0 = 10
            x1 = 16+6
            y1 = 10+3
            isOutside = False
            for agent in self.world.agents:
                # Check if the agent is outside the room
                # Get agent position
                isWithinBounds = agent.isWithinLocationBounds(x0, y0, x1, y1)
                if (isWithinBounds == False):
                    isOutside = True
                    break

            if (isOutside == True):
                self.scorecardAgentOutside.updateScore(1, True, associatedUUIDs=[], associatedNotes="The agent is outside the shed")


        # Update score
        score = 0
        maxScore = 0
        for element in self.scoreCard:
            score += element.score
            maxScore += element.maxScore
        self.score = score
        self.maxScore = maxScore

        # Check whether the task is complete
        # Here, the task is complete if the key is not rusted, and the agent is outside
        if (self.scorecardKeyNotRusted.completed and self.scorecardAgentOutside.completed):
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)
        else:
            self.completed = False
            self.completedSuccessfully = False


#
#   Specific Task: Rusted key task (Challenge)
#
class RustedKeyTaskChallenge(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "You were venturing into the wilderness of Planet X to a storage shed to fetch some supplies. The shed door locked behind you, and your key appears too rusted to work in the lock. "
        taskDescription += "You remember one of the other colonists mentioning that some combination of the chemicals in the shed could make a rust remover, but you can't remember the details. "
        taskDescription += "You need to figure out a way to remove the rust from the key, and use it to open the shed door, and make your way back to the colony. "
        taskDescription += "You have no means of communication with the other colonists, and must figure this out on your own. \n"
        taskDescription += "Some helpful notes: \n"
        taskDescription += "1. The rusted key is in the shed. \n"
        taskDescription += "2. The shed door is locked, and will only open using a non-rusted key. \n"
        taskDescription += "3. You can mix chemicals in the jar. Just use a chemical dispenser on the jar to add chemicals to the jar. \n"
        taskDescription += "4. Chemicals placed in the same jar will automatically mix, though it may take a step or two for this to happen. \n"
        taskDescription += "5. Placing the rusted key into the jar will automatically apply the chemicals to the key.  If the chemical mixture is correct, the rust will be removed, though it may take a step or two. \n"
        taskDescription += "6. If you need to clean the jar of chemicals to try a new combination, use the bottle washer on the jar. \n"
        taskDescription += "7. When you have successfully derusted the key and opened the door, please leave the shed. \n"
        ## taskDescription += "HERE'S A WALKTHROUGH/HINT: To complete this task, you need to pick up the jar, put 1 unit of Chemical A and 2 units of Chemical C into the jar, and then put the key into the jar.  The key will change from rusted to not rusted.  Then you can open the door, walk 3 steps out, and the task will be completed."

        Task.__init__(self, "RustedKeyTaskChallenge", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 2                       # Maximum score

        #scoringInfo['dispensers'] = [dispenser1, dispenser2, dispenser3]
        #scoringInfo['mixingJar'] = mixingJar
        #scoringInfo['bottleCleaner'] = BottleCleaner
        #scoringInfo['key'] = rustyKey

        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardKey = ScorecardElement("Take key", "The key has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardKey)
        self.scorecardJar = ScorecardElement("Take jar", "The mixing jar has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardJar)

        # Has used each chemical dispenser
        self.scorecardUsedDispensers = ScorecardElement("Use each chemical dispenser", "Each chemical dispenser has been used", maxScore=3)
        self.scoreCard.append(self.scorecardUsedDispensers)

        # Has used the bottle cleaner
        self.scorecardUsedBottleCleaner = ScorecardElement("Use bottle cleaner", "The bottle cleaner has been used", maxScore=1)
        self.scoreCard.append(self.scorecardUsedBottleCleaner)

        # Has a mixture containing more than one chemical in the jar
        self.scorecardMixtureInJar = ScorecardElement("Mix chemicals in jar", "The jar contains a mixture of chemicals", maxScore=1)
        self.scoreCard.append(self.scorecardMixtureInJar)

        # Has placed the key in the jar
        self.scorecardKeyInJar = ScorecardElement("Place key in jar", "The key has been placed in the jar at least once", maxScore=1)
        self.scoreCard.append(self.scorecardKeyInJar)

        # Key is not rusted
        self.scorecardKeyNotRusted = ScorecardElement("Key is not rusted", "The key is not rusted (includes levels for reduced amounts of rust)", maxScore=3)
        self.scoreCard.append(self.scorecardKeyNotRusted)

        # Agent is outside
        self.scorecardAgentOutside = ScorecardElement("Agent is outside", "The agent is outside the shed", maxScore=1)
        self.scoreCard.append(self.scorecardAgentOutside)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check whether key has been in agents inventory
        if (not self.scorecardKey.completed):
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.type == "agent"):
                self.scorecardKey.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been in an agent's (UUID: " + str(key.parentContainer.uuid) + ") inventory")

        # Check whether jar has been in agents inventory
        if (not self.scorecardJar.completed):
            jar = self.scoringInfo['mixingJar']
            if (jar.parentContainer != None) and (jar.parentContainer.type == "agent"):
                self.scorecardJar.updateScore(1, True, associatedUUIDs=[jar.uuid, jar.parentContainer.uuid], associatedNotes="The jar has been in an agent's (UUID: " + str(jar.parentContainer.uuid) + ") inventory")

        # Check whether each chemical dispenser has been used
        if (not self.scorecardUsedDispensers.completed):
            usedDispensers = set()
            for agent in self.world.agents:
                for dispenser in self.scoringInfo['dispensers']:
                    # query the action history
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=dispenser, arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                    if (len(foundActions) > 0):
                        usedDispensers.add(dispenser.uuid)

            isCompleted = False
            if (len(usedDispensers) == 3):
                isCompleted = True
            associatedNotes = str(len(usedDispensers)) + " of " + str(len(self.scoringInfo["dispensers"])) + " dispensers have been used"
            self.scorecardUsedDispensers.updateScore(len(usedDispensers), isCompleted, associatedUUIDs=list(usedDispensers), associatedNotes=associatedNotes)

        # Check whether the bottle cleaner has been used
        if (not self.scorecardUsedBottleCleaner.completed):
            usedBottleCleaner = False
            for agent in self.world.agents:
                foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo['bottleCleaner'], arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                if (len(foundActions) > 0):
                    usedBottleCleaner = True
                    break
            if (usedBottleCleaner == True):
                self.scorecardUsedBottleCleaner.updateScore(1, usedBottleCleaner, associatedUUIDs=[self.scoringInfo['bottleCleaner'].uuid], associatedNotes="The bottle cleaner has been used")

        # Check whether the jar contains a mixture of chemicals
        if (not self.scorecardMixtureInJar.completed):
            # Check to see whether the jar contains a substance, and if the substance has a mixture (mixtureDict)
            for cObj in self.scoringInfo['mixingJar'].contents:
                if (cObj.type == "substance"):
                    mixtureSubstances = cObj.attributes['mixtureDict'].keys()
                    if (len(mixtureSubstances) > 1):
                        self.scorecardMixtureInJar.updateScore(1, True, associatedUUIDs=[cObj.uuid], associatedNotes="The mixing jar has contained two or more chemicals in a mixture (" + str(", ".join(mixtureSubstances)) + ")")
                        break

        # Check whether the key is in the jar
        if (not self.scorecardKeyInJar.completed):
            # Check whether the key's parent container is the jar
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.uuid == self.scoringInfo['mixingJar'].uuid):
                self.scorecardKeyInJar.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been placed in the jar at least once")

        # Check whether the key is not rusted
        if (not self.scorecardKeyNotRusted.completed):
            # Check the rust level of the key
            key = self.scoringInfo['key']
#        self.attributes['isRusted'] = isRusted                    # Is the key rusted?
#        self.attributes['rustLevel'] = 3 if isRusted else 0       # Description of the rust (0=none, 1=light, 2=medium, 3=heavy)
            if (key.attributes['isRusted'] == False):
                self.scorecardKeyNotRusted.updateScore(3, True, associatedUUIDs=[key.uuid], associatedNotes="The key is not rusted")
            else:
                # Partial
                rustScore = 3 - key.attributes['rustLevel']
                if (rustScore == 3):
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is fully rusted")
                else:
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is partially rusted")

        # Check to see whether one or more agents are outside of the room
        if (not self.scorecardAgentOutside.completed):
            # Bounds
            x0 = 16
            y0 = 10
            x1 = 16+7
            y1 = 10+3
            isOutside = False
            for agent in self.world.agents:
                # Check if the agent is outside the room
                # Get agent position
                isWithinBounds = agent.isWithinLocationBounds(x0, y0, x1, y1)
                if (isWithinBounds == False):
                    isOutside = True
                    break

            if (isOutside == True):
                self.scorecardAgentOutside.updateScore(1, True, associatedUUIDs=[], associatedNotes="The agent is outside the shed")


        # Update score
        score = 0
        maxScore = 0
        for element in self.scoreCard:
            score += element.score
            maxScore += element.maxScore
        self.score = score
        self.maxScore = maxScore

        # Check whether the task is complete
        # Here, the task is complete if the key is not rusted, and the agent is outside
        if (self.scorecardKeyNotRusted.completed and self.scorecardAgentOutside.completed):
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)
        else:
            self.completed = False
            self.completedSuccessfully = False


#
#   Specific Task: Rusted key task
#
class RustedKeyTaskEasy(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "You were venturing into the wilderness of Planet X to a storage shed to fetch some supplies. The shed door locked behind you, and your key appears too rusted to work in the lock. "
        taskDescription += "You remember one of the other colonists mentioning that one of the chemicals in the shed is a rust remover, but unfortunately the chemicals you see are all unlabelled."
        taskDescription += "You need to figure out which is a rust remover, and use it to remove the rust from the key.  Then, use the key to open the shed door, and make your way back to the colony. "
        taskDescription += "You have no means of communication with the other colonists, and must figure this out on your own. \n"
        taskDescription += "Some helpful notes: \n"
        taskDescription += "1. The rusted key is in the shed. \n"
        taskDescription += "2. The shed door is locked, and will only open using a non-rusted key. \n"
        taskDescription += "3. You can add chemicals to the jar. Just use a chemical dispenser on the jar to add chemicals to the jar. \n"
        taskDescription += "4. The solution is only a single chemical, and should not require mixing chemicals. If you add multiple chemicals into the same jar, they will automatically mix, having different properties than any single chemical. \n"
        taskDescription += "5. Placing the rusted key into the jar will automatically apply the chemical in that jar to the key.  If the chemical is correct, the rust will be removed, though it may up to 2 steps for this to happen. \n"
        taskDescription += "6. If you need to clean the jar of chemicals to try a new chemical, use the bottle washer on the jar. \n"
        taskDescription += "7. When you have successfully derusted the key and opened the door, please leave the shed. \n"
        ## taskDescription += "HERE'S A WALKTHROUGH/HINT: To complete this task, you need to pick up the jar, put 1 unit of Chemical A and 2 units of Chemical C into the jar, and then put the key into the jar.  The key will change from rusted to not rusted.  Then you can open the door, walk 3 steps out, and the task will be completed."

        Task.__init__(self, "RustedKeyTaskEasy", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 2                       # Maximum score

        #scoringInfo['dispensers'] = [dispenser1, dispenser2, dispenser3]
        #scoringInfo['mixingJar'] = mixingJar
        #scoringInfo['bottleCleaner'] = BottleCleaner
        #scoringInfo['key'] = rustyKey

        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardKey = ScorecardElement("Take key", "The key has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardKey)
        self.scorecardJar = ScorecardElement("Take jar", "The mixing jar has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardJar)

        # Has used each chemical dispenser
        self.scorecardUsedDispensers = ScorecardElement("Use each chemical dispenser", "Each chemical dispenser has been used", maxScore=4)
        self.scoreCard.append(self.scorecardUsedDispensers)

        # Has used the bottle cleaner
        self.scorecardUsedBottleCleaner = ScorecardElement("Use bottle cleaner", "The bottle cleaner has been used", maxScore=1)
        self.scoreCard.append(self.scorecardUsedBottleCleaner)

        # Has placed the key in the jar
        self.scorecardKeyInJar = ScorecardElement("Place key in jar", "The key has been placed in the jar at least once", maxScore=1)
        self.scoreCard.append(self.scorecardKeyInJar)

        # Key is not rusted
        self.scorecardKeyNotRusted = ScorecardElement("Key is not rusted", "The key is not rusted (includes levels for reduced amounts of rust)", maxScore=3)
        self.scoreCard.append(self.scorecardKeyNotRusted)

        # Agent is outside
        self.scorecardAgentOutside = ScorecardElement("Agent is outside", "The agent is outside the shed", maxScore=1)
        self.scoreCard.append(self.scorecardAgentOutside)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check whether key has been in agents inventory
        if (not self.scorecardKey.completed):
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.type == "agent"):
                self.scorecardKey.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been in an agent's (UUID: " + str(key.parentContainer.uuid) + ") inventory")

        # Check whether jar has been in agents inventory
        if (not self.scorecardJar.completed):
            jar = self.scoringInfo['mixingJar']
            if (jar.parentContainer != None) and (jar.parentContainer.type == "agent"):
                self.scorecardJar.updateScore(1, True, associatedUUIDs=[jar.uuid, jar.parentContainer.uuid], associatedNotes="The jar has been in an agent's (UUID: " + str(jar.parentContainer.uuid) + ") inventory")

        # Check whether each chemical dispenser has been used
        if (not self.scorecardUsedDispensers.completed):
            usedDispensers = set()
            for agent in self.world.agents:
                for dispenser in self.scoringInfo['dispensers']:
                    # query the action history
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=dispenser, arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                    if (len(foundActions) > 0):
                        usedDispensers.add(dispenser.uuid)

            isCompleted = False
            if (len(usedDispensers) >= 4):
                isCompleted = True
            associatedNotes = str(len(usedDispensers)) + " of " + str(len(self.scoringInfo["dispensers"])) + " dispensers have been used"
            self.scorecardUsedDispensers.updateScore(len(usedDispensers), isCompleted, associatedUUIDs=list(usedDispensers), associatedNotes=associatedNotes)

        # Check whether the bottle cleaner has been used
        if (not self.scorecardUsedBottleCleaner.completed):
            usedBottleCleaner = False
            for agent in self.world.agents:
                foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo['bottleCleaner'], arg2=self.scoringInfo['mixingJar'], stopAtFirst=True)
                if (len(foundActions) > 0):
                    usedBottleCleaner = True
                    break
            if (usedBottleCleaner == True):
                self.scorecardUsedBottleCleaner.updateScore(1, usedBottleCleaner, associatedUUIDs=[self.scoringInfo['bottleCleaner'].uuid], associatedNotes="The bottle cleaner has been used")

        # Check whether the key is in the jar
        if (not self.scorecardKeyInJar.completed):
            # Check whether the key's parent container is the jar
            key = self.scoringInfo['key']
            if (key.parentContainer != None) and (key.parentContainer.uuid == self.scoringInfo['mixingJar'].uuid):
                self.scorecardKeyInJar.updateScore(1, True, associatedUUIDs=[key.uuid, key.parentContainer.uuid], associatedNotes="The key has been placed in the jar at least once")

        # Check whether the key is not rusted
        if (not self.scorecardKeyNotRusted.completed):
            # Check the rust level of the key
            key = self.scoringInfo['key']
#        self.attributes['isRusted'] = isRusted                    # Is the key rusted?
#        self.attributes['rustLevel'] = 3 if isRusted else 0       # Description of the rust (0=none, 1=light, 2=medium, 3=heavy)
            if (key.attributes['isRusted'] == False):
                self.scorecardKeyNotRusted.updateScore(3, True, associatedUUIDs=[key.uuid], associatedNotes="The key is not rusted")
            else:
                # Partial
                rustScore = 3 - key.attributes['rustLevel']
                if (rustScore == 3):
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is fully rusted")
                else:
                    self.scorecardKeyNotRusted.updateScore(rustScore, completed=False, associatedUUIDs=[key.uuid], associatedNotes="The key is partially rusted")

        # Check to see whether one or more agents are outside of the room
        if (not self.scorecardAgentOutside.completed):
            # Bounds
            x0 = 16
            y0 = 10
            x1 = 16+6
            y1 = 10+3
            isOutside = False
            for agent in self.world.agents:
                # Check if the agent is outside the room
                # Get agent position
                isWithinBounds = agent.isWithinLocationBounds(x0, y0, x1, y1)
                if (isWithinBounds == False):
                    isOutside = True
                    break

            if (isOutside == True):
                self.scorecardAgentOutside.updateScore(1, True, associatedUUIDs=[], associatedNotes="The agent is outside the shed")


        # Update score
        score = 0
        maxScore = 0
        for element in self.scoreCard:
            score += element.score
            maxScore += element.maxScore
        self.score = score
        self.maxScore = maxScore

        # Check whether the task is complete
        # Here, the task is complete if the key is not rusted, and the agent is outside
        if (self.scorecardKeyNotRusted.completed and self.scorecardAgentOutside.completed):
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)
        else:
            self.completed = False
            self.completedSuccessfully = False


#
#   Specific Task: Archaeology dig task
#
class ArchaeologyDigEasy(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "You are in archaeological dig store room on Planet X.  3 ancient artifacts have been found. "
        taskDescription += "Your task is to date the artifacts with the radiocarbon meter.  Then, once completed, place the red flag directly beside (i.e. one square right/to the west) of the artifact with the oldest age."

        Task.__init__(self, "ArchaeologyDigTaskEasy", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 1                       # Maximum score
        self.uncoveredArtifacts = set()

        self.scorecardRadiocarbonMeter = ScorecardElement("Take radiocarbon meter", "The radiocarbon meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardRadiocarbonMeter)
        self.scorecardFlag = ScorecardElement("Take flag", "The flag has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardFlag)

        self.scorecardArtifactsDated = ScorecardElement("Artifacts dated", "The artifacts have been dated with the radiocarbon meter", maxScore=3)
        self.scoreCard.append(self.scorecardArtifactsDated)

        self.scorecardFlagPlaced = ScorecardElement("Flag placed", "The flag has been placed in the correct location", maxScore=1)
        self.scoreCard.append(self.scorecardFlagPlaced)

        # Critical Hypotheses
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # scoringInfo["unknownArtifacts"] = []
    # scoringInfo["signs"] = []
    # scoringInfo["targetSign"] = sign
    # scoringInfo["radioCarbonMeter"] = radioCarbonMeter
    # scoringInfo["shovel"] = shovel
    # scoringInfo["flag"] = flag


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check if they have the radioisotope meter in an agent's inventory
        if (not self.scorecardRadiocarbonMeter.completed):
            radioisotopeMeterContainer = self.scoringInfo["radioCarbonMeter"].parentContainer
            if (radioisotopeMeterContainer != None):
                if (radioisotopeMeterContainer.type == "agent"):
                    self.scorecardRadiocarbonMeter.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["radioCarbonMeter"].uuid], associatedNotes="The radiocarbon meter has been in the inventory of the agent with uuid " + str(self.scoringInfo["radioCarbonMeter"].uuid))

        # Check if they have the flag in an agent's inventory
        if (not self.scorecardFlag.completed):
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer != None):
                if (flagContainer.type == "agent"):
                    self.scorecardFlag.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid], associatedNotes="The flag has been in the inventory of the agent with uuid " + str(flagContainer.uuid))

        # Check if the radioisotope meter has been used on 3 seed artifacts
        if (not self.scorecardArtifactsDated.completed):
            seedArtifactsDated = set()
            for agent in self.world.getUserAgents():
                for artifact in self.scoringInfo["unknownArtifacts"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["radioCarbonMeter"], arg2=artifact, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        seedArtifactsDated.add(artifact.uuid)

            numArtifactsDated = len(seedArtifactsDated)
            isComplete = False
            if (numArtifactsDated >= 3):
                isComplete = True
            self.scorecardArtifactsDated.updateScore(score=numArtifactsDated, completed=isComplete, associatedUUIDs=list(seedArtifactsDated), associatedNotes="The following artifacts have been dated: " + str(seedArtifactsDated))


        # Check if the flag has been placed near ANY of the signs (+/- 2 grid spaces).
        # NOTE, changed this so that the flag must be plaged at (x+1, y) of one of the artifacts.
        # `targetArtifact` is the correct one.
        if (not self.scorecardFlagPlaced.completed):
            # First, check to see if the flag has been placed
            flagPlaced = False
            placedCorrectly = False
            objectUUID = None
            if (self.scoringInfo["flag"].parentContainer == None):
                flagLocation = self.scoringInfo["flag"].getWorldLocation()  # (x, y) tuple
                for artifact in self.scoringInfo["unknownArtifacts"]:
                    artifactLocation = artifact.getWorldLocation()
                    # Check to see if the flag has been placed one square to the right of the artifact
                    if (flagLocation[0] == artifactLocation[0]+1) and (flagLocation[1] == artifactLocation[1]):
                        flagPlaced = True
                        objectUUID = artifact.uuid
                        if (artifact == self.scoringInfo["targetArtifact"]):
                            placedCorrectly = True
                        break

            # Update the scorecard
            if (flagPlaced == True):
                if (placedCorrectly == True):
                    self.scorecardFlagPlaced.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, objectUUID], associatedNotes="The flag has been placed near the correct artifact")
                else:
                    self.scorecardFlagPlaced.updateScore(score=0, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, objectUUID], associatedNotes="The flag has been placed near an incorrect artifact")

            # If the flag has been placed, the task is complete
            if (flagPlaced == True):
                self.completed = True
                if (placedCorrectly == True):
                    self.completedSuccessfully = True
                else:
                    self.completedSuccessfully = False
            else:
                self.completed = False
                self.completedSuccessfully = False

        # Update the score, as the sum of the scorecard elements
        self.score = 0
        self.maxScore = 0
        for element in self.scoreCard:
            self.score += element.score
            self.maxScore += element.maxScore

#
#   Archaeology Dig Task (Normal)
#
class ArchaeologyDigNormal(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        taskDescription = "You are on an archaeological dig on Planet X.  3 ancient sites have been found. "
        taskDescription += "Your task is to excavate the sites, and date any artifacts with the radiocarbon meter.  Then, once completed, place the red flag beside the sign of the dig site with the oldest artifact. "

        Task.__init__(self, "ArchaeologyDigTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 1                       # Maximum score
        self.uncoveredArtifacts = set()

        self.scorecardRadiocarbonMeter = ScorecardElement("Take radiocarbon meter", "The radiocarbon meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardRadiocarbonMeter)
        self.scorecardShovel = ScorecardElement("Take shovel", "The shovel has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardShovel)
        self.scorecardFlag = ScorecardElement("Take flag", "The flag has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardFlag)

        self.scorecardArtifactsUncovered = ScorecardElement("Artifacts uncovered", "The artifacts have been uncovered", maxScore=3)
        self.scoreCard.append(self.scorecardArtifactsUncovered)

        self.scorecardArtifactsDated = ScorecardElement("Artifacts dated", "The artifacts have been dated with the radiocarbon meter", maxScore=3)
        self.scoreCard.append(self.scorecardArtifactsDated)

        self.scorecardFlagPlaced = ScorecardElement("Flag placed", "The flag has been placed in the correct location", maxScore=1)
        self.scoreCard.append(self.scorecardFlagPlaced)

        # Critical Hypotheses
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # scoringInfo["unknownArtifacts"] = []
    # scoringInfo["signs"] = []
    # scoringInfo["targetSign"] = sign
    # scoringInfo["radioCarbonMeter"] = radioCarbonMeter
    # scoringInfo["shovel"] = shovel
    # scoringInfo["flag"] = flag


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check if they have the radioisotope meter in an agent's inventory
        if (not self.scorecardRadiocarbonMeter.completed):
            radioisotopeMeterContainer = self.scoringInfo["radioCarbonMeter"].parentContainer
            if (radioisotopeMeterContainer != None):
                if (radioisotopeMeterContainer.type == "agent"):
                    self.scorecardRadiocarbonMeter.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["radioCarbonMeter"].uuid], associatedNotes="The radiocarbon meter has been in the inventory of the agent with uuid " + str(self.scoringInfo["radioCarbonMeter"].uuid))

        # Check if they have the shovel in an agent's inventory
        if (not self.scorecardShovel.completed):
            shovelContainer = self.scoringInfo["shovel"].parentContainer
            if (shovelContainer != None):
                if (shovelContainer.type == "agent"):
                    self.scorecardShovel.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["shovel"].uuid], associatedNotes="The shovel has been in the inventory of the agent with uuid " + str(shovelContainer.uuid))

        # Check if they have the flag in an agent's inventory
        if (not self.scorecardFlag.completed):
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer != None):
                if (flagContainer.type == "agent"):
                    self.scorecardFlag.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid], associatedNotes="The flag has been in the inventory of the agent with uuid " + str(flagContainer.uuid))

        # Check if the 3 unknown objects have been uncovered
        if (not self.scorecardArtifactsUncovered.completed):
            # Measure the number of unknown artifacts that have been uncovered
            for artifact in self.scoringInfo["unknownArtifacts"]:
                parentContainer = artifact.parentContainer
                if (parentContainer != None):
                    if (parentContainer.type == "soil"):
                        if (parentContainer.attributes["hasHole"] == True):
                            self.uncoveredArtifacts.add(artifact.uuid)
                    else:
                        # If not in soil, then it's been moved (which means it must have been uncovered)
                        self.uncoveredArtifacts.add(artifact.uuid)

            numArtifactsUncovered = len(self.uncoveredArtifacts)

            # Update the scorecard
            isComplete = False
            if (numArtifactsUncovered >= 3):
                isComplete = True
            self.scorecardArtifactsUncovered.updateScore(score=numArtifactsUncovered, completed=isComplete, associatedUUIDs=list(self.uncoveredArtifacts), associatedNotes="The following artifacts have been uncovered: " + str(self.uncoveredArtifacts))


        # Check if the radioisotope meter has been used on 3 seed artifacts
        if (not self.scorecardArtifactsDated.completed):
            seedArtifactsDated = set()
            for agent in self.world.getUserAgents():
                for artifact in self.scoringInfo["unknownArtifacts"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["radioCarbonMeter"], arg2=artifact, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        seedArtifactsDated.add(artifact.uuid)

            numArtifactsDated = len(seedArtifactsDated)
            isComplete = False
            if (numArtifactsDated >= 3):
                isComplete = True
            self.scorecardArtifactsDated.updateScore(score=numArtifactsDated, completed=isComplete, associatedUUIDs=list(seedArtifactsDated), associatedNotes="The following artifacts have been dated: " + str(seedArtifactsDated))


        # Check if the flag has been placed near ANY of the signs (+/- 2 grid spaces).
        if (not self.scorecardFlagPlaced.completed):
            # First, check to see if the flag has been placed
            flagPlaced = False
            placedCorrectly = False
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer == None) or (flagContainer.attributes["isAgent"] == False):
                for sign in self.scoringInfo["signs"]:
                    distance = sign.distanceTo(self.scoringInfo["flag"])
                    if (distance <= 2):
                        flagPlaced = True
                        placedNearSignUUID = sign.uuid
                        # Check if the flag has been placed near the correct sign
                        if (sign.uuid == self.scoringInfo["targetSign"].uuid):
                            placedCorrectly = True
                        break

            # Update the scorecard
            if (flagPlaced == True):
                if (placedCorrectly == True):
                    self.scorecardFlagPlaced.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near the correct sign")
                else:
                    self.scorecardFlagPlaced.updateScore(score=0, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near an incorrect sign")

            # If the flag has been placed, the task is complete
            if (flagPlaced == True):
                self.completed = True
                if (placedCorrectly == True):
                    self.completedSuccessfully = True
                else:
                    self.completedSuccessfully = False
            else:
                self.completed = False
                self.completedSuccessfully = False

        # Update the score, as the sum of the scorecard elements
        self.score = 0
        self.maxScore = 0
        for element in self.scoreCard:
            self.score += element.score
            self.maxScore += element.maxScore

#
#   Specific Task: Archaeology dig task (generic radioisotopes)
#
class ArchaeologyDigGenericRadioisotopes(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        taskDescription = "You are on an archaeological dig on Planet X.  6 sites of suspected ancient artifacts have been found, 3 of which have already been uncovered. "
        taskDescription += "It's not clear how or if radioisotope dating works on Planet X, or how it would differ from Earth, but your task is to figure out if it can be used. "
        taskDescription += "Your task is to excavate the remaining sites, and figure out a way to use the radioisotope meter to approximately date the artifacts.  Then, once completed, place the red flag beside the sign of the dig site with the oldest artifact. "

        Task.__init__(self, "ArchaeologyDigTaskGenericRadioisotope", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 1                       # Maximum score

        self.uncoveredArtifacts = set()         # A list of the artifacts that have been uncovered


        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardRadioisotopeMeter = ScorecardElement("Take radioisotope meter", "The radioisotope meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardRadioisotopeMeter)
        self.scorecardShovel = ScorecardElement("Take shovel", "The shovel has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardShovel)
        self.scorecardFlag = ScorecardElement("Take flag", "The flag has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardFlag)

        # 3 unknown objects uncovered
        self.scorecardUncoveredArtifacts = ScorecardElement("Uncover 3 unknown artifacts", "3 unknown artifacts have been uncovered", maxScore=3)
        self.scoreCard.append(self.scorecardUncoveredArtifacts)

        # Radioisotope meter used on 3 seed artifacts
        self.scorecardRadioisotopeMeterUsedSeed = ScorecardElement("Use radioisotope meter (seed)", "The radioisotope meter has been used on 3 seed artifacts", maxScore=3)
        self.scoreCard.append(self.scorecardRadioisotopeMeterUsedSeed)

        # Radioisotope meter used on 3 unknown artifacts
        self.scorecardRadioisotopeMeterUsedUnknown = ScorecardElement("Use radioisotope meter on (unknown)", "The radioisotope meter has been used on 3 unknown artifacts", maxScore=3)
        self.scoreCard.append(self.scorecardRadioisotopeMeterUsedUnknown)

        # Flag moved to correct location (or not) -- ends task
        self.scorecardFlagPlaced = ScorecardElement("Move flag to correct location", "The flag has been moved to the correct location", maxScore=1)
        self.scoreCard.append(self.scorecardFlagPlaced)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Scoring Info passed from the scenario
        # scoringInfo["seedArtifacts"] = []
        # scoringInfo["unknownArtifacts"] = []
        # scoringInfo["radioisotopeMeter"] = radioisotopeMeter
        # scoringInfo["shovel"] = shovel
        # scoringInfo["flag"] = flag
        # scoringInfo["signs"] = []
        # scoringInfo["targetSign"]  # The oldest sign, where the flag should be placed

    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check if they have the radioisotope meter in an agent's inventory
        if (not self.scorecardRadioisotopeMeter.completed):
            radioisotopeMeterContainer = self.scoringInfo["radioisotopeMeter"].parentContainer
            if (radioisotopeMeterContainer != None):
                if (radioisotopeMeterContainer.type == "agent"):
                    self.scorecardRadioisotopeMeter.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["radioisotopeMeter"].uuid], associatedNotes="The radioisotope meter has been in the inventory of the agent with uuid " + str(self.scoringInfo["radioisotopeMeter"].uuid))

        # Check if they have the shovel in an agent's inventory
        if (not self.scorecardShovel.completed):
            shovelContainer = self.scoringInfo["shovel"].parentContainer
            if (shovelContainer != None):
                if (shovelContainer.type == "agent"):
                    self.scorecardShovel.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["shovel"].uuid], associatedNotes="The shovel has been in the inventory of the agent with uuid " + str(shovelContainer.uuid))

        # Check if they have the flag in an agent's inventory
        if (not self.scorecardFlag.completed):
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer != None):
                if (flagContainer.type == "agent"):
                    self.scorecardFlag.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid], associatedNotes="The flag has been in the inventory of the agent with uuid " + str(flagContainer.uuid))

        # Check if the 3 unknown objects have been uncovered
        if (not self.scorecardUncoveredArtifacts.completed):
            # Measure the number of unknown artifacts that have been uncovered
            for artifact in self.scoringInfo["unknownArtifacts"]:
                parentContainer = artifact.parentContainer
                if (parentContainer != None):
                    if (parentContainer.type == "soil"):
                        if (parentContainer.attributes["hasHole"] == True):
                            self.uncoveredArtifacts.add(artifact.uuid)
                    else:
                        # If not in soil, then it's been moved (which means it must have been uncovered)
                        self.uncoveredArtifacts.add(artifact.uuid)

            numArtifactsUncovered = len(self.uncoveredArtifacts)

            # Update the scorecard
            isComplete = False
            if (numArtifactsUncovered >= 3):
                isComplete = True
            self.scorecardUncoveredArtifacts.updateScore(score=numArtifactsUncovered, completed=isComplete, associatedUUIDs=list(self.uncoveredArtifacts), associatedNotes="The following artifacts have been uncovered: " + str(self.uncoveredArtifacts))


        # Check if the radioisotope meter has been used on 3 seed artifacts
        if (not self.scorecardRadioisotopeMeterUsedSeed.completed):
            seedArtifactsDated = set()
            for agent in self.world.getUserAgents():
                for artifact in self.scoringInfo["seedArtifacts"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["radioisotopeMeter"], arg2=artifact, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        seedArtifactsDated.add(artifact.uuid)

            numArtifactsDated = len(seedArtifactsDated)
            isComplete = False
            if (numArtifactsDated >= 3):
                isComplete = True
            self.scorecardRadioisotopeMeterUsedSeed.updateScore(score=numArtifactsDated, completed=isComplete, associatedUUIDs=list(seedArtifactsDated), associatedNotes="The following seed artifacts have been dated: " + str(seedArtifactsDated))

        # Check if the radioisotope meter has been used on 3 unknown artifacts
        if (not self.scorecardRadioisotopeMeterUsedUnknown.completed):
            unknownArtifactsDated = set()
            for agent in self.world.getUserAgents():
                for artifact in self.scoringInfo["unknownArtifacts"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["radioisotopeMeter"], arg2=artifact, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        unknownArtifactsDated.add(artifact.uuid)

            numArtifactsDated = len(unknownArtifactsDated)
            isComplete = False
            if (numArtifactsDated >= 3):
                isComplete = True
            self.scorecardRadioisotopeMeterUsedUnknown.updateScore(score=numArtifactsDated, completed=isComplete, associatedUUIDs=list(unknownArtifactsDated), associatedNotes="The following unknown artifacts have been dated: " + str(unknownArtifactsDated))


        # Check if the flag has been placed near ANY of the signs (+/- 2 grid spaces).
        if (not self.scorecardFlagPlaced.completed):
            # First, check to see if the flag has been placed
            flagPlaced = False
            placedCorrectly = False
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer == None) or (flagContainer.attributes["isAgent"] == False):
                for sign in self.scoringInfo["signs"]:
                    distance = sign.distanceTo(self.scoringInfo["flag"])
                    if (distance <= 2):
                        flagPlaced = True
                        placedNearSignUUID = sign.uuid
                        # Check if the flag has been placed near the correct sign
                        if (sign.uuid == self.scoringInfo["targetSign"].uuid):
                            placedCorrectly = True
                        break

            # Update the scorecard
            if (flagPlaced == True):
                if (placedCorrectly == True):
                    self.scorecardFlagPlaced.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near the correct sign")
                else:
                    self.scorecardFlagPlaced.updateScore(score=0, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near an incorrect sign")

            # If the flag has been placed, the task is complete
            if (flagPlaced == True):
                self.completed = True
                if (placedCorrectly == True):
                    self.completedSuccessfully = True
                else:
                    self.completedSuccessfully = False
            else:
                self.completed = False
                self.completedSuccessfully = False

        # Update the score, as the sum of the scorecard elements
        self.score = 0
        self.maxScore = 0
        for element in self.scoreCard:
            self.score += element.score
            self.maxScore += element.maxScore



#
#   Specific Task: Soil Nutrient Task
#
class SoilNutrientTaskNormal(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        # ["potassium", "titanium", "lithium", "thorium", "barium"]
        taskDescription = "You are at a botanical research station on Planet X.  A species of plant has been identified that appears to grow very quickly in the presence of an unusual nutrient uncommon on Earth. "
        taskDescription += "Other scientists have narrowed down the nutrient to be one of the following: Potassium, Titanium, Lithium, Thorium, or Barium. "
        taskDescription += "Your task is to figure out which nutrient it is, and what specific amount of the nutrient (low, medium, or high) is required in the soil for the plant to grow. "
        taskDescription += "\n\nTo support your work, a pilot field was set up with 12 plots of soil, each with a different combination of nutrients.  The pilot field is located to the south west part of the research station. "
        taskDescription += "The research station is equipped with three test fields, where you can configure the nutrient levels in the field using the nearby soil nutrient controller. Once you configure the nutrients for a field, it *can't* be changed again. "
        taskDescription += "Under the right conditions, the plant tends to grow very quickly, so you should be able to see the results of your work within a few steps. "
        taskDescription += "Inside the storage facility are some tools that may be helpful for your work, including a soil nutrient meter, a jar of seeds, and a shovel. "
        taskDescription += "To plant the seeds, dig a hole in the soil, place a seed in the hole, then put the soil back into the hole.  If the conditions are correct, the plant will grow from the seed. "
        taskDescription += "As part of your discovery process, you should grow at least 2 new plants to maturity. "

        Task.__init__(self, "SoilNutrientTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 6                       # Maximum score

        self.newSeedsPlanted = set()            # New seeds planted since the start of the task
        self.newPlantsGrown = set()             # New plants grown since the start of the task

        # Scorecard elements
        self.scorecardSoilNutrientMeterPresent = ScorecardElement("Soil Nutrient Meter Present", "The soil nutrient meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)
        self.scorecardShovelPresent = ScorecardElement("Shovel Present", "The shovel has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardShovelPresent)
        self.scorecardJarPresent = ScorecardElement("Seed Jar Present", "The seed jar has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardJarPresent)
        self.scorecardUseSoilNutrientMeter = ScorecardElement("Use Soil Nutrient Meter", "The soil nutrient meter has been used on at least 2 squares of soil from the pilot field", maxScore=2)
        self.scoreCard.append(self.scorecardUseSoilNutrientMeter)
        self.scorecardAtLeastTwoSeedsPlanted = ScorecardElement("At Least 2 Seeds Planted", "At least 2 seeds have been planted in the test fields", maxScore=2)
        self.scoreCard.append(self.scorecardAtLeastTwoSeedsPlanted)
        self.scorecardAtLeastTwoNewPlants = ScorecardElement("At Least Two New Plants", "At least two new plants (mushrooms) have been grown to maturity in the test fields", maxScore=2)
        self.scoreCard.append(self.scorecardAtLeastTwoNewPlants)

        # TODO: Add subtask that requires the agent to set the nutrients of at least one field


        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]
        self.testSoilTiles = scoringInfo["testSoilTiles"]
        # Scoring Info passed from the scenario
        # scoringInfo["startingPlants"] = []
        # scoringInfo["startingSeeds"].append(seed)
        # scoringInfo["soilNutrientMeter"] = obj
        # scoringInfo["shovel"] = obj
        # scoringInfo["jar"] = obj            # Seed jar
        # scoringInfo["pilotFieldSoilTiles"]


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardSoilNutrientMeterPresent.completed):
            soilNutrientMeterContainer = self.scoringInfo["soilNutrientMeter"].parentContainer
            if (soilNutrientMeterContainer != None):
                if (soilNutrientMeterContainer.type == "agent"):
                    self.scorecardSoilNutrientMeterPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["soilNutrientMeter"].uuid], associatedNotes="The soil nutrient meter has been in the inventory of the agent with uuid " + str(soilNutrientMeterContainer.uuid))
        #self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)

        # Check if the shovel is in an agent's inventory
        if (not self.scorecardShovelPresent.completed):
            shovelContainer = self.scoringInfo["shovel"].parentContainer
            if (shovelContainer != None):
                if (shovelContainer.type == "agent"):
                    self.scorecardShovelPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["shovel"].uuid], associatedNotes="The shovel has been in the inventory of the agent with uuid " + str(shovelContainer.uuid))
        #self.scoreCard.append(self.scorecardShovelPresent)

        # Check if the jar is in an agent's inventory
        if (not self.scorecardJarPresent.completed):
            jarContainer = self.scoringInfo["jar"].parentContainer
            if (jarContainer != None):
                if (jarContainer.type == "agent"):
                    self.scorecardJarPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["jar"].uuid], associatedNotes="The jar has been in the inventory of the agent with uuid " + str(jarContainer.uuid))
        #self.scoreCard.append(self.scorecardJarPresent)

        # Check if the soil nutrient meter has been used on at least 2 squares of soil from the pilot field
        if (not self.scorecardUseSoilNutrientMeter.completed):
            soilTilesChecked = set()
            for agent in self.world.getUserAgents():
                for soilTile in self.scoringInfo["pilotFieldSoilTiles"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["soilNutrientMeter"], arg2=soilTile, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        soilTilesChecked.add(soilTile.uuid)

            numSoilTilesChecked = len(soilTilesChecked)
            isComplete = False
            if (numSoilTilesChecked >= 2):
                isComplete = True
            self.scorecardUseSoilNutrientMeter.updateScore(score=numSoilTilesChecked, completed=isComplete, associatedUUIDs=list(soilTilesChecked), associatedNotes="The following soil tiles have been checked with the soil nutrient meter: " + str(soilTilesChecked))

        # Check for at least one seed that is not a starting seed to be in the ground
        # First, get all objects
        allObjects = self.world.getAllWorldObjects()

        if (not self.scorecardAtLeastTwoSeedsPlanted.completed):
            # Get locations of test soil tiles
            testSoilTileLocations = []
            for testSoilTile in self.scoringInfo["testSoilTiles"]:
                testSoilTileLocations.append(testSoilTile.getWorldLocation())

            for obj in allObjects:
                if (obj.type == "seed"):
                    # Make sure this seed isn't in the list of starting seeds
                    if (obj not in self.scoringInfo["startingSeeds"]):
                        # Check if the seed is in the ground
                        parentContainer = obj.parentContainer
                        if (parentContainer != None):
                            if (parentContainer.type == "soil"):
                                # Make sure there's no hole in the soil (i.e. the hole is filled in/the seed is planted)
                                if (parentContainer.attributes['hasHole'] == False):
                                    # Check to see if the location of this object is one of the test soil tiles
                                    isOnTestSoilTile = False
                                    for testSoilTileLocation in testSoilTileLocations:
                                        print("Checking if " + str(obj.getWorldLocation()) + " is on " + str(testSoilTileLocation))
                                        if (obj.getWorldLocation() == testSoilTileLocation):
                                            isOnTestSoilTile = True
                                            print("\tMATCH")
                                            break
                                    if (isOnTestSoilTile):
                                        self.newSeedsPlanted.add(obj.uuid)

            numNewSeedsPlanted = len(self.newSeedsPlanted)
            isCompleted = False
            if (numNewSeedsPlanted >= 2):
                isCompleted = True
                numNewSeedsPlanted = 2
            self.scorecardAtLeastTwoSeedsPlanted.updateScore(score=numNewSeedsPlanted, completed=isCompleted, associatedUUIDs=list(self.newSeedsPlanted), associatedNotes="At least two new seeds have been planted in the ground")


        # Check for at least 2 new plants (Mushrooms) to exist
        if (not self.scorecardAtLeastTwoNewPlants.completed):
            # Get locations of test soil tiles
            testSoilTileLocations = []
            for testSoilTile in self.scoringInfo["testSoilTiles"]:
                testSoilTileLocations.append(testSoilTile.getWorldLocation())

            # Check for new plants, that weren't in the simulation when it was initialized
            for obj in allObjects:
                if (isinstance(obj, Mushroom)):
                    if (obj not in self.scoringInfo["startingPlants"]):
                        if ("locationGrown" in obj.attributes):
                            grownOnTestLocation = False
                            for testSoilTileLocation in testSoilTileLocations:
                                if (obj.attributes["locationGrown"] == testSoilTileLocation):
                                    grownOnTestLocation = True
                                    break
                            if (grownOnTestLocation):
                                self.newPlantsGrown.add(obj)

            # Count the number of new plants
            numMaturePlants = len(self.newPlantsGrown)
            completedTwoNewPlants = False
            if (numMaturePlants >= 2):
                numMaturePlants = 2
                completedTwoNewPlants = True

            self.scorecardAtLeastTwoNewPlants.updateScore(score=numMaturePlants, completed=completedTwoNewPlants, associatedUUIDs=[plant.uuid for plant in self.newPlantsGrown], associatedNotes="At least two new plants have been grown to maturity")

            # Ultimately, it's having successfully grown 2 new plants that determines whether the task is complete
            if (completedTwoNewPlants == True):
                self.completed = True
                self.completedSuccessfully = True


        # if (not self.scorecardAtLeastTwoSeedsPlanted.completed):
        #     for obj in allObjects:
        #         if (obj.type == "seed"):
        #             # Make sure this seed isn't in the list of starting seeds
        #             if (obj not in self.scoringInfo["startingSeeds"]):
        #                 # Check if the seed is in the ground
        #                 parentContainer = obj.parentContainer
        #                 if (parentContainer != None):
        #                     # Make sure the parent container is one of the test soil tiles
        #                     isOnTestSoilTile = False
        #                     for testSoilTile in self.scoringInfo["pilotFieldSoilTiles"]:
        #                         if (parentContainer.uuid == testSoilTile.uuid):
        #                             isOnTestSoilTile = True
        #                             break
        #                     if (isOnTestSoilTile):
        #                         # Make sure there's no hole in the soil (i.e. the hole is filled in/the seed is planted)
        #                         if (parentContainer.attributes['hasHole'] == False):
        #                             self.newSeedsPlanted.add(obj.uuid)
        #     numNewSeedsPlanted = len(self.newSeedsPlanted)
        #     isCompleted = False
        #     if (numNewSeedsPlanted >= 2):
        #         isCompleted = True
        #     self.scorecardAtLeastTwoSeedsPlanted.updateScore(score=numNewSeedsPlanted, completed=isCompleted, associatedUUIDs=list(self.newSeedsPlanted), associatedNotes="At least two new seeds have been planted in the ground")


        # # Check for at least 2 new plants (Mushrooms) to exist
        # if (not self.scorecardAtLeastTwoNewPlants.completed):
        #     # Check for new plants, that weren't in the simulation when it was initialized
        #     for obj in allObjects:
        #         if (isinstance(obj, Mushroom)):
        #             if (obj not in self.scoringInfo["startingPlants"]):
        #                 # Make sure the parent container is one of the test soil tiles
        #                 parentContainer = obj.parentContainer
        #                 isOnTestSoilTile = False
        #                 for testSoilTile in self.scoringInfo["pilotFieldSoilTiles"]:
        #                     if (parentContainer != None) and(parentContainer.uuid == testSoilTile.uuid):
        #                         isOnTestSoilTile = True
        #                         break
        #                 if (isOnTestSoilTile):
        #                     self.newPlantsGrown.add(obj)

        #     # Count the number of new plants
        #     numMaturePlants = len(self.newPlantsGrown)
        #     completedTwoNewPlants = False
        #     if (numMaturePlants >= 2):
        #         numMaturePlants = 2
        #         completedTwoNewPlants = True

        #     self.scorecardAtLeastTwoNewPlants.updateScore(score=numMaturePlants, completed=completedTwoNewPlants, associatedUUIDs=[plant.uuid for plant in self.newPlantsGrown], associatedNotes="At least two new plants have been grown to maturity")

        #     # Ultimately, it's having successfully grown 2 new plants that determines whether the task is complete
        #     if (completedTwoNewPlants == True):
        #         self.completed = True
        #         self.completedSuccessfully = True

        #self.scoreCard.append(self.scorecardAtLeastTwoNewPlants)


        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore

        # TODO: Add a timeout that marks the task complete 100 steps after the last soil controller has been used?


#
#   Specific Task: Soil Nutrient Task
#
class SoilNutrientTaskEasy(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        # ["potassium", "titanium", "lithium", "thorium", "barium"]
        taskDescription = "You are at a botanical research station on Planet X.  The plants on Planet X only appear to grow in the presence of an unusual nutrient uncommon on Earth. "
        taskDescription += "Other scientists have narrowed down the nutrient to be one of the following: Potassium, Titanium, Lithium, Thorium, or Barium. "
        taskDescription += "Your task is to figure out which nutrient it is, and inter this answer into the Soil Controller computer. "
        taskDescription += "To support your research, a pilot field was set up with 6 plots of soil, each containing a randomly chosen nutrient. "
        taskDescription += "The research station is also equipped with a soil nutrient meter. "

        Task.__init__(self, "SoilNutrientTaskEasy", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 6                       # Maximum score

        self.newSeedsPlanted = set()            # New seeds planted since the start of the task
        self.newPlantsGrown = set()             # New plants grown since the start of the task

        # Scorecard elements
        self.scorecardSoilNutrientMeterPresent = ScorecardElement("Soil Nutrient Meter Present", "The soil nutrient meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)
        self.scorecardUseSoilNutrientMeter = ScorecardElement("Use Soil Nutrient Meter", "The soil nutrient meter has been used on at least 2 squares of soil from the pilot field", maxScore=2)
        self.scoreCard.append(self.scorecardUseSoilNutrientMeter)
        #self.scorecardAtLeastTwoNewPlants = ScorecardElement("At Least Two New Plants", "At least two new plants (mushrooms) have been grown to maturity in the test fields", maxScore=2)
        #self.scoreCard.append(self.scorecardAtLeastTwoNewPlants)
        self.scorecardSelectCorrectNutrient = ScorecardElement("Select Correct Nutrient", "The correct nutrient has been selected", maxScore=1)
        self.scoreCard.append(self.scorecardSelectCorrectNutrient)

        ## TODO: Add subtask to check for correct answer


        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardSoilNutrientMeterPresent.completed):
            soilNutrientMeterContainer = self.scoringInfo["soilNutrientMeter"].parentContainer
            if (soilNutrientMeterContainer != None):
                if (soilNutrientMeterContainer.type == "agent"):
                    self.scorecardSoilNutrientMeterPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["soilNutrientMeter"].uuid], associatedNotes="The soil nutrient meter has been in the inventory of the agent with uuid " + str(soilNutrientMeterContainer.uuid))
        #self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)

        # Check if the soil nutrient meter has been used on at least 2 squares of soil from the pilot field
        if (not self.scorecardUseSoilNutrientMeter.completed):
            soilTilesChecked = set()
            for agent in self.world.getUserAgents():
                for soilTile in self.scoringInfo["pilotFieldSoilTiles"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["soilNutrientMeter"], arg2=soilTile, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        soilTilesChecked.add(soilTile.uuid)

            numSoilTilesChecked = len(soilTilesChecked)
            isComplete = False
            if (numSoilTilesChecked >= 2):
                isComplete = True
            self.scorecardUseSoilNutrientMeter.updateScore(score=numSoilTilesChecked, completed=isComplete, associatedUUIDs=list(soilTilesChecked), associatedNotes="The following soil tiles have been checked with the soil nutrient meter: " + str(soilTilesChecked))


        ## TODO: Add subtask to check for correct answer
        if (not self.scorecardSelectCorrectNutrient.completed):
            # Check for signals
            signalCorrect = "select_" + self.scoringInfo["positiveNutrient"]
            signalsIncorrect = []
            for nutrient in self.scoringInfo["negativeNutrients"]:
                signalsIncorrect.append("select_" + nutrient)

            # Check for any signals
            soilController = self.scoringInfo["soilController"]

            # Check for the correct signal
            if (soilController.hasState(signalCorrect) == True):
                self.scorecardSelectCorrectNutrient.updateScore(score=1, completed=True, associatedUUIDs=[soilController.uuid], associatedNotes="The correct nutrient has been selected")
            else:
                # Check for incorrect signals
                for signal in signalsIncorrect:
                    if (soilController.hasState(signal)):
                        self.scorecardSelectCorrectNutrient.updateScore(score=0, completed=True, associatedUUIDs=[soilController.uuid], associatedNotes="An incorrect nutrient has been selected")


        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore

        # Task completion
        if (self.scorecardSelectCorrectNutrient.completed):
            self.completed = True
            if (self.scorecardSelectCorrectNutrient.score == 1):
                self.completedSuccessfully = True
            else:
                self.completedSuccessfully = False




#
#   Specific Task: Soil Nutrient Task
#
class SoilNutrientTaskChallenge(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        # ["potassium", "titanium", "lithium", "thorium", "barium"]
        taskDescription = "You are at a botanical research station on Planet X.  A species of plant has been identified that appears to grow very quickly in the presence of unusual nutrient(s) uncommon on Earth. "
        taskDescription += "Other scientists have narrowed down the nutrient list to include the following: Potassium, Titanium, Lithium, Thorium, or Barium. "
        taskDescription += "Your task is to figure out what nutrient or combination of nutrients make the plants grow, including what specific amount of the nutrient(s) (i.e. low, medium, or high) is required in the soil for the plant to grow. "
        taskDescription += "\n\nTo support your work, a pilot field was set up with 18 plots of soil, each with a different combination of nutrients.  The pilot field is located to the south west part of the research station. "
        taskDescription += "The research station is equipped with three test fields, where you can configure the nutrient levels in the field using the nearby soil nutrient controller. Once you configure the nutrients for a field, it *can't* be changed again. "
        taskDescription += "Under the right conditions, the plant tends to grow very quickly, so you should be able to see the results of your work within a few steps. "
        taskDescription += "Inside the storage facility are some tools that may be helpful for your work, including a soil nutrient meter, a jar of seeds, and a shovel. "
        taskDescription += "To plant the seeds, dig a hole in the soil, place a seed in the hole, then put the soil back into the hole.  If the conditions are correct, the plant will grow from the seed. "
        taskDescription += "As part of your discovery process, you should grow at least 2 new plants to maturity. "

        Task.__init__(self, "SoilNutrientTaskChallenge", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 6                       # Maximum score

        self.newSeedsPlanted = set()            # New seeds planted since the start of the task
        self.newPlantsGrown = set()             # New plants grown since the start of the task

        # Scorecard elements
        self.scorecardSoilNutrientMeterPresent = ScorecardElement("Soil Nutrient Meter Present", "The soil nutrient meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)
        self.scorecardShovelPresent = ScorecardElement("Shovel Present", "The shovel has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardShovelPresent)
        self.scorecardJarPresent = ScorecardElement("Seed Jar Present", "The seed jar has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardJarPresent)
        self.scorecardUseSoilNutrientMeter = ScorecardElement("Use Soil Nutrient Meter", "The soil nutrient meter has been used on at least 2 squares of soil from the pilot field", maxScore=2)
        self.scoreCard.append(self.scorecardUseSoilNutrientMeter)
        self.scorecardAtLeastTwoSeedsPlanted = ScorecardElement("At Least 2 Seeds Planted", "At least 2 seeds have been planted in the test fields", maxScore=2)
        self.scoreCard.append(self.scorecardAtLeastTwoSeedsPlanted)
        self.scorecardAtLeastTwoNewPlants = ScorecardElement("At Least Two New Plants", "At least two new plants (mushrooms) have been grown to maturity in the test fields", maxScore=2)
        self.scoreCard.append(self.scorecardAtLeastTwoNewPlants)

        # TODO: Add subtask that requires the agent to set the nutrients of at least one field


        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]
        self.testSoilTiles = scoringInfo["testSoilTiles"]
        # Scoring Info passed from the scenario
        # scoringInfo["startingPlants"] = []
        # scoringInfo["startingSeeds"].append(seed)
        # scoringInfo["soilNutrientMeter"] = obj
        # scoringInfo["shovel"] = obj
        # scoringInfo["jar"] = obj            # Seed jar
        # scoringInfo["pilotFieldSoilTiles"]


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardSoilNutrientMeterPresent.completed):
            soilNutrientMeterContainer = self.scoringInfo["soilNutrientMeter"].parentContainer
            if (soilNutrientMeterContainer != None):
                if (soilNutrientMeterContainer.type == "agent"):
                    self.scorecardSoilNutrientMeterPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["soilNutrientMeter"].uuid], associatedNotes="The soil nutrient meter has been in the inventory of the agent with uuid " + str(soilNutrientMeterContainer.uuid))
        #self.scoreCard.append(self.scorecardSoilNutrientMeterPresent)

        # Check if the shovel is in an agent's inventory
        if (not self.scorecardShovelPresent.completed):
            shovelContainer = self.scoringInfo["shovel"].parentContainer
            if (shovelContainer != None):
                if (shovelContainer.type == "agent"):
                    self.scorecardShovelPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["shovel"].uuid], associatedNotes="The shovel has been in the inventory of the agent with uuid " + str(shovelContainer.uuid))
        #self.scoreCard.append(self.scorecardShovelPresent)

        # Check if the jar is in an agent's inventory
        if (not self.scorecardJarPresent.completed):
            jarContainer = self.scoringInfo["jar"].parentContainer
            if (jarContainer != None):
                if (jarContainer.type == "agent"):
                    self.scorecardJarPresent.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["jar"].uuid], associatedNotes="The jar has been in the inventory of the agent with uuid " + str(jarContainer.uuid))
        #self.scoreCard.append(self.scorecardJarPresent)

        # Check if the soil nutrient meter has been used on at least 2 squares of soil from the pilot field
        if (not self.scorecardUseSoilNutrientMeter.completed):
            soilTilesChecked = set()
            for agent in self.world.getUserAgents():
                for soilTile in self.scoringInfo["pilotFieldSoilTiles"]:
                    foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["soilNutrientMeter"], arg2=soilTile, stopAtFirst=True)
                    if (len(foundActions) > 0):
                        soilTilesChecked.add(soilTile.uuid)

            numSoilTilesChecked = len(soilTilesChecked)
            isComplete = False
            if (numSoilTilesChecked >= 2):
                isComplete = True
            self.scorecardUseSoilNutrientMeter.updateScore(score=numSoilTilesChecked, completed=isComplete, associatedUUIDs=list(soilTilesChecked), associatedNotes="The following soil tiles have been checked with the soil nutrient meter: " + str(soilTilesChecked))

        # Check for at least one seed that is not a starting seed to be in the ground
        # First, get all objects
        allObjects = self.world.getAllWorldObjects()

        if (not self.scorecardAtLeastTwoSeedsPlanted.completed):
            # Get locations of test soil tiles
            testSoilTileLocations = []
            for testSoilTile in self.scoringInfo["testSoilTiles"]:
                testSoilTileLocations.append(testSoilTile.getWorldLocation())

            for obj in allObjects:
                if (obj.type == "seed"):
                    # Make sure this seed isn't in the list of starting seeds
                    if (obj not in self.scoringInfo["startingSeeds"]):
                        # Check if the seed is in the ground
                        parentContainer = obj.parentContainer
                        if (parentContainer != None):
                            if (parentContainer.type == "soil"):
                                # Make sure there's no hole in the soil (i.e. the hole is filled in/the seed is planted)
                                if (parentContainer.attributes['hasHole'] == False):
                                    # Check to see if the location of this object is one of the test soil tiles
                                    isOnTestSoilTile = False
                                    for testSoilTileLocation in testSoilTileLocations:
                                        print("Checking if " + str(obj.getWorldLocation()) + " is on " + str(testSoilTileLocation))
                                        if (obj.getWorldLocation() == testSoilTileLocation):
                                            isOnTestSoilTile = True
                                            print("\tMATCH")
                                            break
                                    if (isOnTestSoilTile):
                                        self.newSeedsPlanted.add(obj.uuid)

            numNewSeedsPlanted = len(self.newSeedsPlanted)
            isCompleted = False
            if (numNewSeedsPlanted >= 2):
                isCompleted = True
                numNewSeedsPlanted = 2
            self.scorecardAtLeastTwoSeedsPlanted.updateScore(score=numNewSeedsPlanted, completed=isCompleted, associatedUUIDs=list(self.newSeedsPlanted), associatedNotes="At least two new seeds have been planted in the ground")


        # Check for at least 2 new plants (Mushrooms) to exist
        if (not self.scorecardAtLeastTwoNewPlants.completed):
            # Get locations of test soil tiles
            testSoilTileLocations = []
            for testSoilTile in self.scoringInfo["testSoilTiles"]:
                testSoilTileLocations.append(testSoilTile.getWorldLocation())

            # Check for new plants, that weren't in the simulation when it was initialized
            for obj in allObjects:
                if (isinstance(obj, Mushroom)):
                    if (obj not in self.scoringInfo["startingPlants"]):
                        if ("locationGrown" in obj.attributes):
                            grownOnTestLocation = False
                            for testSoilTileLocation in testSoilTileLocations:
                                if (obj.attributes["locationGrown"] == testSoilTileLocation):
                                    grownOnTestLocation = True
                                    break
                            if (grownOnTestLocation):
                                self.newPlantsGrown.add(obj)

            # Count the number of new plants
            numMaturePlants = len(self.newPlantsGrown)
            completedTwoNewPlants = False
            if (numMaturePlants >= 2):
                numMaturePlants = 2
                completedTwoNewPlants = True

            self.scorecardAtLeastTwoNewPlants.updateScore(score=numMaturePlants, completed=completedTwoNewPlants, associatedUUIDs=[plant.uuid for plant in self.newPlantsGrown], associatedNotes="At least two new plants have been grown to maturity")

            # Ultimately, it's having successfully grown 2 new plants that determines whether the task is complete
            if (completedTwoNewPlants == True):
                self.completed = True
                self.completedSuccessfully = True

        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore

        # TODO: Add a timeout that marks the task complete 100 steps after the last soil controller has been used?




#
#   Specific Task: Reactor Lab (Normal)
#
class ReactorTaskNormal(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        taskDescription = "You are at the Quantum Reactor Lab on Planet X. "
        taskDescription += "Quantum Crystals offer the potential to generate a great deal of power, but require their respective crystal reactors to be tuned to a specific frequency that appears unique for each crystal. "
        taskDescription += "Through great effort, a previous research scientist manually stumbled upon the correct frequencies for Crystal 1 and Crystal 2, which are in their respective reactors. "
        taskDescription += "Your task is to use science to figure out the correct frequency for Crystal 3 and Crystal 4, set their reactors to the appropriate frequency, and install the crystals. "
        taskDescription += "Once all four crystals are installed, the reactor will be able to generate a great deal of power. "
        taskDescription += "To support your task, a number of scientific instruments are available in the lab. "

        Task.__init__(self, "ReactorTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 6                       # Maximum score

        self.crystalsTaken = set()              # Crystals that have been taken
        self.instrumentsUsed = set()            # Instruments that have been used
        self.crystalsExamined = set()           # Crystals that have been examined
        self.crystalsExaminedByCriticalInstrument = set()  # Crystals that have been examined by the critical instrument
        self.reactorsChanged = set()            # Reactors that have had their values changed

        # Scorecard elements
        self.scorecardQuantumCrystalsPresent = ScorecardElement("Quantum Crystals Taken", "The quantum crystals have each been in an agent's inventory", maxScore=4)
        self.scoreCard.append(self.scorecardQuantumCrystalsPresent)

        # Each instrument has been used with at least one crystal
        self.scorecardInstrumentsUsed = ScorecardElement("Instruments Used", "Each scientific instrument has been used with at least one crystal", maxScore=5)
        self.scoreCard.append(self.scorecardInstrumentsUsed)

        # Each crystal has been examined by at least one instrument
        self.scorecardCrystalsExamined = ScorecardElement("Crystals Examined", "Each crystal has been examined by at least one instrument", maxScore=4)
        self.scoreCard.append(self.scorecardCrystalsExamined)

        # Each crystal has been examined by the critical instrument required to solve the task
        self.scorecardCriticalInstrumentsUsed = ScorecardElement("Crystals Examined by Critical Instrument", "Each crystal has been examined by the critical instrument", maxScore=4)
        self.scoreCard.append(self.scorecardCriticalInstrumentsUsed)

        # Unknown reactors (3 and 4) have had their values changed
        self.scorecardReactorsChanged = ScorecardElement("Reactors Changed", "The resonance frequency of the unknown reactors have been changed", maxScore=2)
        self.scoreCard.append(self.scorecardReactorsChanged)

        # Unknown reactors (3 and 4) set to correct resonance frequencies
        self.scorecardReactorsSet = ScorecardElement("Reactors Set", "The resonance frequency of the unknown reactors is correct", maxScore=2)
        self.scoreCard.append(self.scorecardReactorsSet)

        # Reactors on
        self.scorecardReactorsOn = ScorecardElement("Reactors On", "The reactors have been successfully activated", maxScore=4)
        self.scoreCard.append(self.scorecardReactorsOn)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

        # Scoring Info passed from the scenario
        # scoringInfo['instruments'] = instruments
        # scoringInfo['quantumCrystals'] = quantumCrystals
        # scoringInfo['reactors'] = crystalReactors
        # scoringInfo['reactorsToChange']
        # scoringInfo['criticalInstrument']


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardQuantumCrystalsPresent.completed):
            for crystal in self.scoringInfo["quantumCrystals"]:
                crystalContainer = crystal.parentContainer
                if (crystalContainer != None):
                    if (crystalContainer.type == "agent"):
                        self.crystalsTaken.add(crystal.uuid)

            numCrystalsTaken = len(self.crystalsTaken)
            isComplete = False
            if (numCrystalsTaken >= 4):
                isComplete = True
            self.scorecardQuantumCrystalsPresent.updateScore(score=numCrystalsTaken, completed=isComplete, associatedUUIDs=list(self.crystalsTaken), associatedNotes="The following quantum crystals have been taken: " + str(self.crystalsTaken))

        # Check if the instruments have been used with at least one crystal
        if (not self.scorecardInstrumentsUsed.completed) or (not self.scorecardCrystalsExamined.completed) or (not self.scorecardCriticalInstrumentsUsed.completed):
            for agent in self.world.getUserAgents():
                for instrument in self.scoringInfo["instruments"]:
                    for crystal in self.scoringInfo["quantumCrystals"]:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2=crystal, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            self.instrumentsUsed.add(instrument.uuid)
                            self.crystalsExamined.add(crystal.uuid)

                            if (instrument.uuid == self.scoringInfo["criticalInstrument"].uuid):
                                self.crystalsExaminedByCriticalInstrument.add(crystal.uuid)

            numInstrumentsUsed = len(self.instrumentsUsed)
            isCompleteInstruments = False
            if (numInstrumentsUsed >= 5):
                isCompleteInstruments = True
            self.scorecardInstrumentsUsed.updateScore(score=numInstrumentsUsed, completed=isCompleteInstruments, associatedUUIDs=list(self.instrumentsUsed), associatedNotes="The following instruments have been used: " + str(self.instrumentsUsed))

            numCrystalsExamined = len(self.crystalsExamined)
            isCompleteCrystals = False
            if (numCrystalsExamined >= 4):
                isCompleteCrystals = True
            self.scorecardCrystalsExamined.updateScore(score=numCrystalsExamined, completed=isCompleteCrystals, associatedUUIDs=list(self.crystalsExamined), associatedNotes="The following crystals have been examined: " + str(self.crystalsExamined))

            numCrystalsExaminedByCriticalInstrument = len(self.crystalsExaminedByCriticalInstrument)
            isCompleteCritical = False
            if (numCrystalsExaminedByCriticalInstrument >= 4):
                isCompleteCritical = True
            self.scorecardCriticalInstrumentsUsed.updateScore(score=numCrystalsExaminedByCriticalInstrument, completed=isCompleteCritical, associatedUUIDs=list(self.crystalsExaminedByCriticalInstrument), associatedNotes="The following crystals have been examined by the critical instrument: " + str(self.crystalsExaminedByCriticalInstrument))


        # Check that the resonance frequency of the unknown (last 2) reactors has been changed from the default value
        if (not self.scorecardReactorsChanged.completed):
            for reactor in self.scoringInfo["reactorsToChange"]:
                if (reactor.attributes["resonanceFreq"] != reactor.attributes["resonanceFreqDefault"]):
                    self.reactorsChanged.add(reactor.uuid)

            numReactorsChanged = len(self.reactorsChanged)
            isComplete = False
            if (numReactorsChanged >= 2):
                isComplete = True
            self.scorecardReactorsChanged.updateScore(score=numReactorsChanged, completed=isComplete, associatedUUIDs=list(self.reactorsChanged), associatedNotes="The following reactors have had their resonance frequency changed: " + str(self.reactorsChanged))

        # Check that the resonance frequency of the unknown (last 2) reactors has been set to the correct value
        numReactorsSet = 0
        unknownCrystals = self.scoringInfo["quantumCrystals"][2:]
        #print("unknown crystals: " + str(unknownCrystals))
        for idx, reactor in enumerate(self.scoringInfo["reactorsToChange"]):
            crystalFreq = unknownCrystals[idx].attributes["resonanceFreq"]
            #if (reactor.attributes["resonanceFreq"] == crystalFreq):
            if (reactor.attributes["isActivated"] == True):
                numReactorsSet += 1

        isComplete = False
        if (numReactorsSet >= 2):
            isComplete = True
        self.scorecardReactorsSet.updateScore(score=numReactorsSet, completed=isComplete, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been set to the correct resonance frequency: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # Check if the reactors have been activated
        #if (not self.scorecardReactorsOn.completed):
        numReactorsActivated = 0
        for reactor in self.scoringInfo["reactors"]:
            if (reactor.attributes["isActivated"] == True):
                numReactorsActivated += 1

        allReactorsOn = False
        if (numReactorsActivated >= 4):
            allReactorsOn = True

        # Score
        self.scorecardReactorsOn.updateScore(score=numReactorsActivated, completed=allReactorsOn, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been activated: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # If this task is complete, then the task is complete
        if (allReactorsOn == True):
            self.completed = True
            self.completedSuccessfully = True
        else:
            self.completed = False
            self.completedSuccessfully = False


        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore


#
#   Specific Task: Reactor Lab (Easy)
#
class ReactorTaskEasy(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        taskDescription = "You are at the Quantum Reactor Lab on Planet X. "
        taskDescription += "Quantum Crystals offer the potential to generate a great deal of power, but require their respective crystal reactors to be tuned to a specific frequency that appears unique for each crystal. "
        taskDescription += "Through great effort, a previous research scientist manually stumbled upon the correct frequencies for Crystal 1 and Crystal 2, which are in their respective reactors. "
        taskDescription += "Your task is to use science to figure out the correct frequency for Crystal 3, set it's reactor to the appropriate frequency, and install the crystal. "
        taskDescription += "Once all three crystals are installed, the reactor will be able to generate a great deal of power. "
        taskDescription += "To support your task, a scientific instrument is available in the lab. "

        Task.__init__(self, "ReactorTaskEasy", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 6                       # Maximum score

        self.crystalsTaken = set()              # Crystals that have been taken
        self.instrumentsUsed = set()            # Instruments that have been used
        self.crystalsExamined = set()           # Crystals that have been examined
        self.crystalsExaminedByCriticalInstrument = set()  # Crystals that have been examined by the critical instrument
        self.reactorsChanged = set()            # Reactors that have had their values changed

        # Scorecard elements
        self.scorecardQuantumCrystalsPresent = ScorecardElement("Quantum Crystals Taken", "The quantum crystals have each been in an agent's inventory", maxScore=3)
        self.scoreCard.append(self.scorecardQuantumCrystalsPresent)

        # Each crystal has been examined by the critical instrument required to solve the task
        self.scorecardCriticalInstrumentsUsed = ScorecardElement("Crystals Examined by Critical Instrument", "Each crystal has been examined by the critical instrument", maxScore=3)
        self.scoreCard.append(self.scorecardCriticalInstrumentsUsed)

        # Unknown reactors (3 and 4) have had their values changed
        self.scorecardReactorsChanged = ScorecardElement("Reactors Changed", "The resonance frequency of the unknown reactors have been changed", maxScore=1)
        self.scoreCard.append(self.scorecardReactorsChanged)

        # Unknown reactors (3 and 4) set to correct resonance frequencies
        self.scorecardReactorsSet = ScorecardElement("Reactors Set", "The resonance frequency of the unknown reactors is correct", maxScore=1)
        self.scoreCard.append(self.scorecardReactorsSet)

        # Reactors on
        self.scorecardReactorsOn = ScorecardElement("Reactors On", "The reactors have been successfully activated", maxScore=3)
        self.scoreCard.append(self.scorecardReactorsOn)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

        # Scoring Info passed from the scenario
        # scoringInfo['instruments'] = instruments
        # scoringInfo['quantumCrystals'] = quantumCrystals
        # scoringInfo['reactors'] = crystalReactors
        # scoringInfo['reactorsToChange']
        # scoringInfo['criticalInstrument']


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardQuantumCrystalsPresent.completed):
            for crystal in self.scoringInfo["quantumCrystals"]:
                crystalContainer = crystal.parentContainer
                if (crystalContainer != None):
                    if (crystalContainer.type == "agent"):
                        self.crystalsTaken.add(crystal.uuid)

            numCrystalsTaken = len(self.crystalsTaken)
            isComplete = False
            if (numCrystalsTaken >= 4):
                isComplete = True
            self.scorecardQuantumCrystalsPresent.updateScore(score=numCrystalsTaken, completed=isComplete, associatedUUIDs=list(self.crystalsTaken), associatedNotes="The following quantum crystals have been taken: " + str(self.crystalsTaken))

        # Check if the instruments have been used with at least one crystal
        if(not self.scorecardCriticalInstrumentsUsed.completed):
            for agent in self.world.getUserAgents():
                for instrument in self.scoringInfo["instruments"]:
                    for crystal in self.scoringInfo["quantumCrystals"]:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2=crystal, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            self.instrumentsUsed.add(instrument.uuid)
                            self.crystalsExamined.add(crystal.uuid)

                            if (instrument.uuid == self.scoringInfo["criticalInstrument"].uuid):
                                self.crystalsExaminedByCriticalInstrument.add(crystal.uuid)

            numCrystalsExaminedByCriticalInstrument = len(self.crystalsExaminedByCriticalInstrument)
            isCompleteCritical = False
            if (numCrystalsExaminedByCriticalInstrument >= 3):
                isCompleteCritical = True
            self.scorecardCriticalInstrumentsUsed.updateScore(score=numCrystalsExaminedByCriticalInstrument, completed=isCompleteCritical, associatedUUIDs=list(self.crystalsExaminedByCriticalInstrument), associatedNotes="The following crystals have been examined by the critical instrument: " + str(self.crystalsExaminedByCriticalInstrument))


        # Check that the resonance frequency of the unknown (last 2) reactors has been changed from the default value
        if (not self.scorecardReactorsChanged.completed):
            for reactor in self.scoringInfo["reactorsToChange"]:
                if (reactor.attributes["resonanceFreq"] != reactor.attributes["resonanceFreqDefault"]):
                    self.reactorsChanged.add(reactor.uuid)

            numReactorsChanged = len(self.reactorsChanged)
            isComplete = False
            if (numReactorsChanged >= 2):
                isComplete = True
            self.scorecardReactorsChanged.updateScore(score=numReactorsChanged, completed=isComplete, associatedUUIDs=list(self.reactorsChanged), associatedNotes="The following reactors have had their resonance frequency changed: " + str(self.reactorsChanged))

        # Check that the resonance frequency of the unknown (last 2) reactors has been set to the correct value
        numReactorsSet = 0
        unknownCrystals = self.scoringInfo["quantumCrystals"][2:]
        #print("unknown crystals: " + str(unknownCrystals))
        for idx, reactor in enumerate(self.scoringInfo["reactorsToChange"]):
            crystalFreq = unknownCrystals[idx].attributes["resonanceFreq"]
            #if (reactor.attributes["resonanceFreq"] == crystalFreq):
            if (reactor.attributes["isActivated"] == True):
                numReactorsSet += 1

        isComplete = False
        if (numReactorsSet >= 2):
            isComplete = True
        self.scorecardReactorsSet.updateScore(score=numReactorsSet, completed=isComplete, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been set to the correct resonance frequency: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # Check if the reactors have been activated
        #if (not self.scorecardReactorsOn.completed):
        numReactorsActivated = 0
        for reactor in self.scoringInfo["reactors"]:
            if (reactor.attributes["isActivated"] == True):
                numReactorsActivated += 1

        allReactorsOn = False
        if (numReactorsActivated >= 3):
            allReactorsOn = True

        # Score
        self.scorecardReactorsOn.updateScore(score=numReactorsActivated, completed=allReactorsOn, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been activated: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # If this task is complete, then the task is complete
        if (allReactorsOn == True):
            self.completed = True
            self.completedSuccessfully = True
        else:
            self.completed = False
            self.completedSuccessfully = False


        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore

#
#   Reactor Task (Challenge)
#
class ReactorTaskChallenge(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        taskDescription = "You are at the Quantum Reactor Lab on Planet X. "
        taskDescription += "Quantum Crystals offer the potential to generate a great deal of power, but require their respective crystal reactors to be tuned to a specific frequency that appears unique for each crystal. "
        taskDescription += "Through great effort, a previous research scientist manually stumbled upon the correct frequencies for Crystal 1, Crystal 2, and Crystal 3, which are in their respective reactors. "
        taskDescription += "Your task is to use science to figure out the correct frequency for Crystal 4 and Crystal 5, set their reactors to the appropriate frequency, and install the crystals. "
        taskDescription += "Once all four crystals are installed, the reactor will be able to generate a great deal of power. "
        taskDescription += "To support your task, a number of scientific instruments are available in the lab. "

        Task.__init__(self, "ReactorTaskChallenge", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 0                       # Maximum score

        self.crystalsTaken = set()              # Crystals that have been taken
        self.instrumentsUsed = set()            # Instruments that have been used
        self.crystalsExamined = set()           # Crystals that have been examined
        self.crystalsExaminedByCriticalInstrument = set()  # Crystals that have been examined by the critical instrument
        self.reactorsChanged = set()            # Reactors that have had their values changed

        # Scorecard elements
        self.scorecardQuantumCrystalsPresent = ScorecardElement("Quantum Crystals Taken", "The quantum crystals have each been in an agent's inventory", maxScore=5)
        self.scoreCard.append(self.scorecardQuantumCrystalsPresent)

        # Each instrument has been used with at least one crystal
        self.scorecardInstrumentsUsed = ScorecardElement("Instruments Used", "Each scientific instrument has been used with at least one crystal", maxScore=5)
        self.scoreCard.append(self.scorecardInstrumentsUsed)

        # Each crystal has been examined by at least one instrument
        self.scorecardCrystalsExamined = ScorecardElement("Crystals Examined", "Each crystal has been examined by at least one instrument", maxScore=5)
        self.scoreCard.append(self.scorecardCrystalsExamined)

        # Each crystal has been examined by the critical instrument required to solve the task
        self.scorecardCriticalInstrumentsUsed = ScorecardElement("Crystals Examined by Critical Instrument", "Each crystal has been examined by the critical instrument", maxScore=5)
        self.scoreCard.append(self.scorecardCriticalInstrumentsUsed)

        # Unknown reactors (3 and 4) have had their values changed
        self.scorecardReactorsChanged = ScorecardElement("Reactors Changed", "The resonance frequency of the unknown reactors have been changed", maxScore=2)
        self.scoreCard.append(self.scorecardReactorsChanged)

        # Unknown reactors (3 and 4) set to correct resonance frequencies
        self.scorecardReactorsSet = ScorecardElement("Reactors Set", "The resonance frequency of the unknown reactors is correct", maxScore=2)
        self.scoreCard.append(self.scorecardReactorsSet)

        # Reactors on
        self.scorecardReactorsOn = ScorecardElement("Reactors On", "The reactors have been successfully activated", maxScore=5)
        self.scoreCard.append(self.scorecardReactorsOn)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

        # Scoring Info passed from the scenario
        # scoringInfo['instruments'] = instruments
        # scoringInfo['quantumCrystals'] = quantumCrystals
        # scoringInfo['reactors'] = crystalReactors
        # scoringInfo['reactorsToChange']
        # scoringInfo['criticalInstrument']


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Clear the previous score and scorecard
        #self.scoreCard = []
        #score = 0

        # Check if they have the soil nutrient meter in an agent's inventory
        if (not self.scorecardQuantumCrystalsPresent.completed):
            for crystal in self.scoringInfo["quantumCrystals"]:
                crystalContainer = crystal.parentContainer
                if (crystalContainer != None):
                    if (crystalContainer.type == "agent"):
                        self.crystalsTaken.add(crystal.uuid)

            numCrystalsTaken = len(self.crystalsTaken)
            isComplete = False
            if (numCrystalsTaken >= 5):
                isComplete = True
            self.scorecardQuantumCrystalsPresent.updateScore(score=numCrystalsTaken, completed=isComplete, associatedUUIDs=list(self.crystalsTaken), associatedNotes="The following quantum crystals have been taken: " + str(self.crystalsTaken))

        # Check if the instruments have been used with at least one crystal
        if (not self.scorecardInstrumentsUsed.completed) or (not self.scorecardCrystalsExamined.completed) or (not self.scorecardCriticalInstrumentsUsed.completed):
            for agent in self.world.getUserAgents():
                for instrument in self.scoringInfo["instruments"]:
                    for crystal in self.scoringInfo["quantumCrystals"]:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=instrument, arg2=crystal, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            self.instrumentsUsed.add(instrument.uuid)
                            self.crystalsExamined.add(crystal.uuid)

                            if (instrument.uuid == self.scoringInfo["criticalInstrument"].uuid):
                                self.crystalsExaminedByCriticalInstrument.add(crystal.uuid)

            numInstrumentsUsed = len(self.instrumentsUsed)
            isCompleteInstruments = False
            if (numInstrumentsUsed >= 5):
                isCompleteInstruments = True
            self.scorecardInstrumentsUsed.updateScore(score=numInstrumentsUsed, completed=isCompleteInstruments, associatedUUIDs=list(self.instrumentsUsed), associatedNotes="The following instruments have been used: " + str(self.instrumentsUsed))

            numCrystalsExamined = len(self.crystalsExamined)
            isCompleteCrystals = False
            if (numCrystalsExamined >= 5):
                isCompleteCrystals = True
            self.scorecardCrystalsExamined.updateScore(score=numCrystalsExamined, completed=isCompleteCrystals, associatedUUIDs=list(self.crystalsExamined), associatedNotes="The following crystals have been examined: " + str(self.crystalsExamined))

            numCrystalsExaminedByCriticalInstrument = len(self.crystalsExaminedByCriticalInstrument)
            isCompleteCritical = False
            if (numCrystalsExaminedByCriticalInstrument >= 5):
                isCompleteCritical = True
            self.scorecardCriticalInstrumentsUsed.updateScore(score=numCrystalsExaminedByCriticalInstrument, completed=isCompleteCritical, associatedUUIDs=list(self.crystalsExaminedByCriticalInstrument), associatedNotes="The following crystals have been examined by the critical instrument: " + str(self.crystalsExaminedByCriticalInstrument))


        # Check that the resonance frequency of the unknown (last 2) reactors has been changed from the default value
        if (not self.scorecardReactorsChanged.completed):
            for reactor in self.scoringInfo["reactorsToChange"]:
                if (reactor.attributes["resonanceFreq"] != reactor.attributes["resonanceFreqDefault"]):
                    self.reactorsChanged.add(reactor.uuid)

            numReactorsChanged = len(self.reactorsChanged)
            isComplete = False
            if (numReactorsChanged >= 2):
                isComplete = True
            self.scorecardReactorsChanged.updateScore(score=numReactorsChanged, completed=isComplete, associatedUUIDs=list(self.reactorsChanged), associatedNotes="The following reactors have had their resonance frequency changed: " + str(self.reactorsChanged))

        # Check that the resonance frequency of the unknown (last 2) reactors has been set to the correct value
        numReactorsSet = 0
        unknownCrystals = self.scoringInfo["quantumCrystals"][3:]
        #print("unknown crystals: " + str(unknownCrystals))
        for idx, reactor in enumerate(self.scoringInfo["reactorsToChange"]):
            crystalFreq = unknownCrystals[idx].attributes["resonanceFreq"]
            #if (reactor.attributes["resonanceFreq"] == crystalFreq):
            if (reactor.attributes["isActivated"] == True):
                numReactorsSet += 1

        isComplete = False
        if (numReactorsSet >= 2):
            isComplete = True
        self.scorecardReactorsSet.updateScore(score=numReactorsSet, completed=isComplete, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been set to the correct resonance frequency: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # Check if the reactors have been activated
        #if (not self.scorecardReactorsOn.completed):
        numReactorsActivated = 0
        for reactor in self.scoringInfo["reactors"]:
            if (reactor.attributes["isActivated"] == True):
                numReactorsActivated += 1

        allReactorsOn = False
        if (numReactorsActivated >= 5):
            allReactorsOn = True

        # Score
        self.scorecardReactorsOn.updateScore(score=numReactorsActivated, completed=allReactorsOn, associatedUUIDs=[reactor.uuid for reactor in self.scoringInfo["reactors"]], associatedNotes="The following reactors have been activated: " + str([reactor.uuid for reactor in self.scoringInfo["reactors"]]))

        # If this task is complete, then the task is complete
        if (allReactorsOn == True):
            self.completed = True
            self.completedSuccessfully = True
        else:
            self.completed = False
            self.completedSuccessfully = False


        # Count the score, based on the scorecard
        score = 0
        maxScore = 0
        for scorecardElement in self.scoreCard:
            score += scorecardElement.score
            maxScore += scorecardElement.maxScore
        self.score = score
        self.maxScore = maxScore


#
#   Specific Task: Rosetta Stone Task
#
class RosettaStoneTask(Task):

    def __init__(self, world, scoringInfo):
        taskDescription = "You find yourself venturing into a small village of Planet X where everyone speaks a dialect unknown to you. An elder (at the center of the village) is trying to tell you something that seems very important. "
        taskDescription += "You need to figure out what the older is saying. \n"
        taskDescription += "Some helpful notes: \n"
        taskDescription += "1. Even though you don't understand them, you can still interact with the inhabitants. \n"
        taskDescription += "2. Look for signs to read, you might be able to decipher some words from them. \n"

        super().__init__("RosettaStoneTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardElder = ScorecardElement("Talk to elder", "The agent has learned about the elder's request.", maxScore=1)
        self.scoreCard.append(self.scorecardElder)

        # Taking critical objects
        self.scorecardGetObjects = ScorecardElement("Collect objects", "Needed objects are in agent's inventory.", maxScore=2)
        self.scoreCard.append(self.scorecardGetObjects)

        # Bring back the objects to the elder
        self.scorecardGiveObjects = ScorecardElement("Give back objects", "Needed objects are in elder's inventory.", maxScore=2)
        self.scoreCard.append(self.scorecardGiveObjects)

        if scoringInfo["learningColor"]:
            self.scorecardColor = ScorecardElement("Visit the paint shop", "The agent visited the paint shop to learn about color.", maxScore=1)
            self.scoreCard.append(self.scorecardColor)

            self.scorecardColorSign = ScorecardElement("Read relevant paint sign", "The paint sign associated to the color has been read.", maxScore=1)
            self.scoreCard.append(self.scorecardColorSign)

        if scoringInfo["learningCount"]:
            self.scorecardCount = ScorecardElement("Visit the school", "The agent visited the school to learn how to count.", maxScore=1)
            self.scoreCard.append(self.scorecardCount)

            # Has used the counting computer
            self.scorecardUseComputer = ScorecardElement("Use counting computer", "The counting computer has been used.", maxScore=1)
            self.scoreCard.append(self.scorecardUseComputer)

            self.scorecardResetComputer = ScorecardElement("Reset counting computer", "The reset card was used on the computer.", maxScore=1)
            self.scoreCard.append(self.scorecardResetComputer)

            # Has used the measuring tape
            self.scorecardMeasuringTape = ScorecardElement("Use measuring tape", "The measuring tape has been used on the flagpole.", maxScore=1)
            self.scoreCard.append(self.scorecardMeasuringTape)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def updateTick(self):
        if self.completed:
            return  # Do not update the score if the task is already marked as completed

        if not self.scorecardElder.completed:
            elder = self.scoringInfo["elder"]
            if "taskGiven" in elder.attributes["states"]:
                self.scorecardElder.updateScore(1, True, associatedUUIDs=[elder.uuid], associatedNotes=f"Agent has talked to the elder (UUID: {elder.uuid}).")

        # Taking critical objects
        if not self.scorecardGetObjects.completed:
            associatedNotes = []
            associatedUUIDs = []

            for agent in self.world.agents:
                if isinstance(agent, NPC):
                    continue

                for obj in agent.getInventory():
                    if self.scoringInfo["item"] in obj.name:
                        if self.scoringInfo["learningColor"] and self.scoringInfo["color"] not in obj.name.split(" "):
                            continue  # Color doesn't match

                        associatedUUIDs.append((obj.uuid, agent.uuid))
                        associatedNotes.append(f"{obj.name} (UUID: {obj.uuid}) is in Agent (UUID: {agent.uuid})'s inventory.")

            if associatedNotes:
                isCompleted = len(associatedNotes) >= self.scoringInfo["count"]
                self.scorecardGetObjects.updateScore(2 if isCompleted else 1, isCompleted, associatedUUIDs, "\n".join(associatedNotes))

        # Give back critical objects to elder
        if not self.scorecardGiveObjects.completed:
            elder = self.scoringInfo["elder"]
            associatedNotes = []
            associatedUUIDs = []
            for obj in elder.getInventory():
                if self.scoringInfo["item"] in obj.name:
                        if self.scoringInfo["learningColor"] and self.scoringInfo["color"] not in obj.name.split(" "):
                            continue  # Color doesn't match

                        associatedUUIDs.append((obj.uuid, elder.uuid))
                        associatedNotes.append(f"{obj.name} (UUID: {obj.uuid}) is in Elder (UUID: {elder.uuid})'s inventory.")

            if associatedNotes:
                isCompleted = len(associatedNotes) >= self.scoringInfo["count"]
                self.scorecardGiveObjects.updateScore(2 if isCompleted else 1, isCompleted, associatedUUIDs, "\n".join(associatedNotes))

        # Check scorecard for learning about counting.
        if self.scoringInfo["learningCount"] and not self.scorecardCount.completed:
            x0, y0, x1, y1 = self.scoringInfo["schoolBounds"]
            associatedUUIDs = []
            for agent in self.world.agents:
                if agent.isWithinLocationBounds(x0, y0, x1, y1):
                    associatedUUIDs.append(agent.uuid)

            if associatedUUIDs:
                self.scorecardCount.updateScore(1, True, associatedUUIDs=associatedUUIDs, associatedNotes=f"Agent (UUID: {', '.join(map(str, associatedUUIDs))}) has visited the school.")

        if self.scoringInfo["learningCount"] and not self.scorecardUseComputer.completed:
            countingComputer = self.scoringInfo["countingComputer"]
            for agent in self.world.agents:
                if agent.actionHistory.queryActionObjects(ActionType.USE, arg1=countingComputer, arg2="*", stopAtFirst=True):
                    self.scorecardUseComputer.updateScore(1, True, associatedUUIDs=[agent.uuid, countingComputer.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has used the counting computer (UUID: {countingComputer.uuid}).")

        if self.scoringInfo["learningCount"] and not self.scorecardResetComputer.completed:
            countingComputer = self.scoringInfo["countingComputer"]
            resetDisk = self.scoringInfo["resetDisk"]
            for agent in self.world.agents:
                if agent.actionHistory.queryActionObjects(ActionType.USE, arg1=countingComputer, arg2=resetDisk, stopAtFirst=True):
                    self.scorecardResetComputer.updateScore(1, True, associatedUUIDs=[agent.uuid, countingComputer.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has used the counting computer (UUID: {countingComputer.uuid}).")

        if self.scoringInfo["learningCount"] and not self.scorecardMeasuringTape.completed:
            flagpole = self.scoringInfo["flagpole"]
            measuringTape = self.scoringInfo["measuringTape"]
            for agent in self.world.agents:
                if agent.actionHistory.queryActionObjects(ActionType.USE, arg1=flagpole, arg2=measuringTape, stopAtFirst=True):
                    self.scorecardMeasuringTape.updateScore(1, True, associatedUUIDs=[agent.uuid, measuringTape.uuid, flagpole.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has used the measuring tape (UUID: {measuringTape.uuid}) on the flagpole (UUID: {flagpole.uuid}).")

        # Check scorecard for learning about color.
        if self.scoringInfo["learningColor"] and not self.scorecardColor.completed:
            x0, y0, x1, y1 = self.scoringInfo["paintShopBounds"]
            associatedUUIDs = []
            for agent in self.world.agents:
                if agent.isWithinLocationBounds(x0, y0, x1, y1):
                    associatedUUIDs.append(agent.uuid)

            if associatedUUIDs:
                self.scorecardColor.updateScore(1, True, associatedUUIDs=associatedUUIDs, associatedNotes=f"Agent (UUID: {', '.join(map(str, associatedUUIDs))}) has visited the school.")

        if self.scoringInfo["learningColor"] and not self.scorecardColorSign.completed:
            colorSign = self.scoringInfo["colorSign"]
            for agent in self.world.agents:
                if agent.actionHistory.queryActionObjects(ActionType.READ, arg1=colorSign, stopAtFirst=True):
                    self.scorecardColorSign.updateScore(1, True, associatedUUIDs=[agent.uuid, colorSign.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has read the relevant color sign (UUID: {colorSign.uuid}).")

        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        self.completed = False
        self.completedSuccessfully = False
        if self.scorecardGiveObjects.completed:# and self.scorecardObjectsReturnedToElder.completed):  # TODO: check for the second condition
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)


class TutorialTask(Task):

    def __init__(self, world, scoringInfo):
        taskDescription = "Where am I?! You just wake up in a strange place... It's discovery time!\n"
        taskDescription += "\nSome helpful notes: \n"
        taskDescription += "1. Feel free to explore your surroundings. \n"
        taskDescription += "2. The elder knows the completion code for the tutorial."

        super().__init__("TutorialTask", taskDescription, world, scoringInfo)

        # Scorecard elements (TODO)
        self.scorecardElder = ScorecardElement("Talk to elder", "The agent has learned about the elder's request.", maxScore=1)
        self.scoreCard.append(self.scorecardElder)

        # Taking critical objects
        self.scorecardGetKey = ScorecardElement("Collect key", "The key is in the agent's inventory.", maxScore=1)
        self.scoreCard.append(self.scorecardGetKey)

        # Read the sign
        # self.scorecardGiveObjects = ScorecardElement("Read the sign", "Needed objects are in elder's inventory.", maxScore=self.scoringInfo["count"])
        # self.scoreCard.append(self.scorecardGiveObjects)

        self.scorecardExit = ScorecardElement("Exit the room", "The agent has left the roo  m.", maxScore=1)
        self.scoreCard.append(self.scorecardExit)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        pass

    def updateTick(self):
        if self.completed:
            return  # Do not update the score if the task is already marked as completed

        if not self.scorecardElder.completed:
            elder = self.scoringInfo["elder"]
            if "taskAccepted" in elder.attributes["states"]:
                self.scorecardElder.updateScore(1, True, associatedUUIDs=[elder.uuid], associatedNotes=f"Agent has talked to the elder (UUID: {elder.uuid}).")

        # Taking critical objects
        if not self.scorecardGetKey.completed:
            associatedNotes = []
            associatedUUIDs = []

            for agent in self.world.agents:
                if isinstance(agent, NPC):
                    continue

                for obj in agent.getInventory():
                    if "key" in obj.name:
                        associatedUUIDs.append((obj.uuid, agent.uuid))
                        associatedNotes.append(f"{obj.name} (UUID: {obj.uuid}) is in Agent (UUID: {agent.uuid})'s inventory.")

            if associatedNotes:
                self.scorecardGetKey.updateScore(1, True, associatedUUIDs, "\n".join(associatedNotes))

        # Check scorecard for leaving the house.
        if not self.scorecardExit.completed:
            x0, y0, x1, y1 = self.scoringInfo["houseBounds"]
            associatedUUIDs = []
            for agent in self.world.agents:
                if not agent.isWithinLocationBounds(x0, y0, x1, y1):
                    associatedUUIDs.append(agent.uuid)

            if associatedUUIDs:
                self.scorecardExit.updateScore(1, True, associatedUUIDs=associatedUUIDs, associatedNotes=f"Agent (UUID: {', '.join(map(str, associatedUUIDs))}) has left the house.")

        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if all the objects have been collected and returned to the elder.
        self.completed = False
        self.completedSuccessfully = False
        if self.scorecardExit.completed:# and self.scorecardObjectsReturnedToElder.completed):  # TODO: check for the second condition
            self.completed = True
            self.completedSuccessfully = True
            #print("Task completed successfully: " + self.taskName)



#
#   Specific Task: Proteomics Task
#

# Normal and Challenge versions of the Proteomics Task use the same class
class ProteomicsTask(Task):
    # Constructor
    def __init__(self, world, scoringInfo, challengeVersion:bool = False):
        # TODO: modify description
        ## TODO: MODIFY DESCRIPTION FOR PROTEOMICS
        taskDescription = "You are in a biological preserve on Planet X, that has 5 different animal species. "
        taskDescription += "We suspect that one of these animal species is not native to the area, but migrated from an isolated island in the recent past. "
        taskDescription += "Your task is to use the proteomics meter to analyze the proteins of each of the 5 animal species, which can be found throughout the environment away from the central statue area, and determine which species is the anomoly. "
        taskDescription += "Once you have completed your task, return to the statue area and drop the red flag directly beside the statue of the animal species that is the anomoly."

        if (challengeVersion == True):
            Task.__init__(self, "ProteomicsTaskChallenge", taskDescription, world, scoringInfo)
        else:
            Task.__init__(self, "ProteomicsTaskNormal", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 1                       # Maximum score

        #self.uncoveredArtifacts = set()         # A list of the artifacts that have been uncovered

        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardProteomicsMeter = ScorecardElement("Take proteomics meter", "The proteomics meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardProteomicsMeter)
        self.scorecardFlag = ScorecardElement("Take flag", "The flag has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardFlag)

        # Proteomics meter used on at least one instance of all 5 animal types
        self.scorecardProteomicsMeterUsedAnimals = ScorecardElement("Use proteomics meter", "The proteomics meter has been used on each of the 5 different animal types", maxScore=5)
        self.scoreCard.append(self.scorecardProteomicsMeterUsedAnimals)

        # Flag moved to correct location (or not) -- ends task
        self.scorecardFlagPlaced = ScorecardElement("Move flag to correct location", "The flag has been moved to the correct location", maxScore=1)
        self.scoreCard.append(self.scorecardFlagPlaced)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check if they have the radioisotope meter in an agent's inventory
        if (not self.scorecardProteomicsMeter.completed):
            meterContainer = self.scoringInfo["meter"].parentContainer
            if (meterContainer != None):
                if (meterContainer.type == "agent"):
                    self.scorecardProteomicsMeter.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["meter"].uuid], associatedNotes="The proteomics meter has been in the inventory of the agent with uuid " + str(self.scoringInfo["meter"].uuid))

        # Check if they have the flag in an agent's inventory
        if (not self.scorecardFlag.completed):
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer != None):
                if (flagContainer.type == "agent"):
                    self.scorecardFlag.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid], associatedNotes="The flag has been in the inventory of the agent with uuid " + str(flagContainer.uuid))

        # Check if the radioisotope meter has been used on 3 seed artifacts
        if (not self.scorecardProteomicsMeterUsedAnimals.completed):
            animalTypesInvestigated = set()
            animalUUIDsInvestigated = set()

            for agent in self.world.getUserAgents():
                for animalIdx in range(0, 5):
                    animalInstances = self.scoringInfo["animal" + str(animalIdx)]
                    for animal in animalInstances:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["meter"], arg2=animal, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            animalUUIDsInvestigated.add(animal.uuid)
                            animalTypesInvestigated.add(animal.name)
                            break

            numAnimalTypesInvestigated = len(animalTypesInvestigated)
            isComplete = False
            if (numAnimalTypesInvestigated >= 5):
                isComplete = True
            self.scorecardProteomicsMeterUsedAnimals.updateScore(score=numAnimalTypesInvestigated, completed=isComplete, associatedUUIDs=list(animalUUIDsInvestigated), associatedNotes="The following animal types have been investigated: " + str(animalTypesInvestigated))


        # Check if the flag has been placed near ANY of the signs (+/- 2 grid spaces).
        if (not self.scorecardFlagPlaced.completed):
            # First, check to see if the flag has been placed
            flagPlaced = False
            placedCorrectly = False
            if (self.scoringInfo["flag"].parentContainer == None):
                for statue in self.scoringInfo["statues"]:
                    distance = statue.distanceTo(self.scoringInfo["flag"])
                    if (distance <= 2):
                        flagPlaced = True
                        placedNearSignUUID = statue.uuid
                        # Check if the flag has been placed near the correct sign
                        if (statue.uuid == self.scoringInfo["correctStatue"].uuid):
                            placedCorrectly = True
                        break

            # Update the scorecard
            if (flagPlaced == True):
                if (placedCorrectly == True):
                    self.scorecardFlagPlaced.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near the correct statue")
                else:
                    self.scorecardFlagPlaced.updateScore(score=0, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near an incorrect statue")

            # If the flag has been placed, the task is complete
            if (flagPlaced == True):
                self.completed = True
                if (placedCorrectly == True):
                    self.completedSuccessfully = True
                else:
                    self.completedSuccessfully = False
            else:
                self.completed = False
                self.completedSuccessfully = False

        # Update the score, as the sum of the scorecard elements
        self.score = 0
        self.maxScore = 0
        for element in self.scoreCard:
            self.score += element.score
            self.maxScore += element.maxScore


# Normal and Challenge versions of the Proteomics Task use the same class
class ProteomicsTaskEasy(Task):
    # Constructor
    def __init__(self, world, scoringInfo):
        # TODO: modify description
        ## TODO: MODIFY DESCRIPTION FOR PROTEOMICS
        taskDescription = "You are in a biological preserve on Planet X, that has 3 different animal species. "
        taskDescription += "We suspect that one of these animal species is not native to the area, but migrated from an isolated island in the recent past. "
        taskDescription += "Your task is to use the proteomics meter to analyze the proteins of each of the 3 animal species, which can be found in the research building you're in, and determine which species is the anomoly. "
        taskDescription += "Once you have completed your task, drop the red flag directly beside (i.e. one square left/to the west) the statue of the animal species that is the anomoly."

        Task.__init__(self, "ProteomicsTaskEasy", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 1                       # Maximum score

        #self.uncoveredArtifacts = set()         # A list of the artifacts that have been uncovered

        # Scorecard elements (TODO)
        # Taking critical objects
        self.scorecardProteomicsMeter = ScorecardElement("Take proteomics meter", "The proteomics meter has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardProteomicsMeter)
        self.scorecardFlag = ScorecardElement("Take flag", "The flag has been in an agent's inventory", maxScore=1)
        self.scoreCard.append(self.scorecardFlag)

        # Proteomics meter used on at least one instance of all 5 animal types
        self.scorecardProteomicsMeterUsedAnimals = ScorecardElement("Use proteomics meter", "The proteomics meter has been used on each of the 3 different animal types", maxScore=3)
        self.scoreCard.append(self.scorecardProteomicsMeterUsedAnimals)

        # Flag moved to correct location (or not) -- ends task
        self.scorecardFlagPlaced = ScorecardElement("Move flag to correct location", "The flag has been moved to the correct location", maxScore=1)
        self.scoreCard.append(self.scorecardFlagPlaced)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]


    # Task setup: Add any necessary objects to the world to perform the task.
    def taskSetup(self):
        # Add the colonists?
        pass

    def initialize(self):
        pass

    # Update the task progress
    def updateTick(self):
        # Do not update the score if the task is already marked as completed
        #if (self.completed == True):
        #    return

        # Check if they have the radioisotope meter in an agent's inventory
        if (not self.scorecardProteomicsMeter.completed):
            meterContainer = self.scoringInfo["meter"].parentContainer
            if (meterContainer != None):
                if (meterContainer.type == "agent"):
                    self.scorecardProteomicsMeter.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["meter"].uuid], associatedNotes="The proteomics meter has been in the inventory of the agent with uuid " + str(self.scoringInfo["meter"].uuid))

        # Check if they have the flag in an agent's inventory
        if (not self.scorecardFlag.completed):
            flagContainer = self.scoringInfo["flag"].parentContainer
            if (flagContainer != None):
                if (flagContainer.type == "agent"):
                    self.scorecardFlag.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid], associatedNotes="The flag has been in the inventory of the agent with uuid " + str(flagContainer.uuid))

        # Check if the radioisotope meter has been used on 3 seed artifacts
        if (not self.scorecardProteomicsMeterUsedAnimals.completed):
            animalTypesInvestigated = set()
            animalUUIDsInvestigated = set()

            for agent in self.world.getUserAgents():
                for animalIdx in range(0, 3):
                    animalInstances = self.scoringInfo["animal" + str(animalIdx)]
                    for animal in animalInstances:
                        foundActions = agent.actionHistory.queryActionObjects(ActionType.USE, arg1=self.scoringInfo["meter"], arg2=animal, stopAtFirst=True)
                        if (len(foundActions) > 0):
                            animalUUIDsInvestigated.add(animal.uuid)
                            animalTypesInvestigated.add(animal.name)
                            break

            numAnimalTypesInvestigated = len(animalTypesInvestigated)
            isComplete = False
            if (numAnimalTypesInvestigated >= 3):
                isComplete = True
            self.scorecardProteomicsMeterUsedAnimals.updateScore(score=numAnimalTypesInvestigated, completed=isComplete, associatedUUIDs=list(animalUUIDsInvestigated), associatedNotes="The following animal types have been investigated: " + str(animalTypesInvestigated))


        # Check if the flag has been placed near ANY of the signs (+/- 2 grid spaces).
        if (not self.scorecardFlagPlaced.completed):
            # First, check to see if the flag has been placed
            flagPlaced = False
            placedCorrectly = False
            if (self.scoringInfo["flag"].parentContainer == None):
                locationFlag = self.scoringInfo["flag"].getWorldLocation()   # Returns a tuple (x, y)
                for statue in self.scoringInfo["statues"]:
                    locationStatue = statue.getWorldLocation()
                    # Here, we're just going to check if the flag is DIRECTLY (1 square) left of the statue (i.e. x-1 of the statue)
                    #if (self.scoringInfo["flag"].x == statue.x - 1) and (self.scoringInfo["flag"].y == statue.y):
                    if (locationFlag[0] == locationStatue[0] - 1) and (locationFlag[1] == locationStatue[1]):
                        flagPlaced = True
                        placedNearSignUUID = statue.uuid
                        # Check if the flag has been placed near the correct sign
                        if (statue.uuid == self.scoringInfo["correctStatue"].uuid):
                            placedCorrectly = True
                        break

                    # distance = statue.distanceTo(self.scoringInfo["flag"])
                    # if (distance <= 1):
                    #     flagPlaced = True
                    #     placedNearSignUUID = statue.uuid
                    #     # Check if the flag has been placed near the correct sign
                    #     if (statue.uuid == self.scoringInfo["correctStatue"].uuid):
                    #         placedCorrectly = True
                    #     break

            # Update the scorecard
            if (flagPlaced == True):
                if (placedCorrectly == True):
                    self.scorecardFlagPlaced.updateScore(score=1, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near the correct statue")
                else:
                    self.scorecardFlagPlaced.updateScore(score=0, completed=True, associatedUUIDs=[self.scoringInfo["flag"].uuid, placedNearSignUUID], associatedNotes="The flag has been placed near an incorrect statue")

            # If the flag has been placed, the task is complete
            if (flagPlaced == True):
                self.completed = True
                if (placedCorrectly == True):
                    self.completedSuccessfully = True
                else:
                    self.completedSuccessfully = False
            else:
                self.completed = False
                self.completedSuccessfully = False

        # Update the score, as the sum of the scorecard elements
        self.score = 0
        self.maxScore = 0
        for element in self.scoreCard:
            self.score += element.score
            self.maxScore += element.maxScore
