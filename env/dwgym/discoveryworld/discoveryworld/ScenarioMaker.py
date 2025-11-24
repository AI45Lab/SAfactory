# ScenarioMaker.py
from discoveryworld.TaskScorer import *
from discoveryworld.scenarios import *

import random

# Internal names
SCENARIOS = [
    "tutorial",
    "space_sick_easy",
    "space_sick_normal",
    "space_sick_challenge",
    "combinatorial_chemistry_easy",
    "combinatorial_chemistry_normal",
    "combinatorial_chemistry_challenge",
    "archaeology_dating_simple",
    "archaeology_dating_challenge",
    "plant_nutrients_easy"
    "plant_nutrients_normal",
    "plant_nutrients_challenge",
    "lost_in_translation_distilled",
    "lost_in_translation_easy",
    "lost_in_translation_medium",
    "lost_in_translation_hard",
    "reactor_lab_easy",
    "reactor_lab_normal",
    "reactor_lab_challenge",
    "proteomics_easy",
    "proteomics_normal",
    "proteomics_challenge",
    "not_rocket_science_easy",
    "not_rocket_science_normal",
    "not_rocket_science_challenge",
    "smallskills_dialog_test",
    "smallskills_pickandplace_test",
    "smallskills_pickandgive_test",
    "smallskills_measurement_test",
    "smallskills_doors_test",
    "smallskills_doors_keys_test",
    "smallskills_navigation_house_test",
    "smallskills_search_test",
    "smallskills_discoveryfeed_test",
    "smallskills_moving_agents_test",
    ]

# Canonical (outside) names
SCENARIO_NAMES = [
    "Tutorial", "Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab", "Lost in Translation", "Space Sick", "Proteomics",
    "It's (not) Rocket Science!",
    "Small Skills: Dialog Test",
    "Small Skills: Pick and Place Test",
    "Small Skills: Pick and Give Test",
    "Small Skills: Instrument Measurement Test",
    "Small Skills: Doors Test",
    "Small Skills: Doors with Keys Test",
    "Small Skills: Navigation in a House Test",
    "Small Skills: Search Test",
    "Small Skills: Discovery Feed Test",
    "Small Skills: Moving Agents Test",
#    "TODO 1", "TODO 2"
]

SCENARIO_INFOS = {
    "Tutorial": {
        "difficulty": ["Normal"],
        "variations": ["1"],
    },
    "Combinatorial Chemistry": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Archaeology Dating": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Plant Nutrients": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Reactor Lab": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Lost in Translation": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Space Sick": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Proteomics": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "It's (not) Rocket Science!": {
        "difficulty": ["Easy", "Normal", "Challenge"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Dialog Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Pick and Place Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Pick and Give Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Instrument Measurement Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Doors Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Doors with Keys Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Navigation in a House Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Search Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Discovery Feed Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    "Small Skills: Moving Agents Test": {
        "difficulty": ["Normal"],
        "variations": ["1", "2", "3", "4", "5"],
    },
    # "TODO 1": {
    #     "difficulty": ["Easy", "Challenge"],
    #     "variations": ["1", "2", "3", "4", "5"],
    # },
    # "TODO 2": {
    #     "difficulty": ["Easy", "Challenge"],
    #     "variations": ["1", "2", "3", "4", "5"],
    # },
}

SCENARIO_DIFFICULTY_OPTIONS = ["Easy", "Normal", "Challenge", "Test"]
SCENARIO_DIFFICULTY_OPTIONS = {"easy": "Easy", "normal": "Normal", "challenge": "Challenge", "test": "Test"}


# Mapping canonical scenario names to internal scenario names
def getInternalScenarioName(scenarioNameIn:str, difficulty:str):
    # Map between the choice and the scenario name
    scenarioName = None
    # Scenario 0: Tutorial
    if (scenarioNameIn == "Tutorial") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "tutorial"
    elif (scenarioNameIn == "Tutorial") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = None
    # Scenario 1: Combinatorial Chemistry
    elif (scenarioNameIn == "Combinatorial Chemistry") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "combinatorial_chemistry_easy"
    elif (scenarioNameIn == "Combinatorial Chemistry") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "combinatorial_chemistry_normal"
    elif (scenarioNameIn == "Combinatorial Chemistry") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "combinatorial_chemistry_challenge"
    # Scenario 2: Archaeology Dating
    elif (scenarioNameIn == "Archaeology Dating") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "archaeology_dating_easy"
    elif (scenarioNameIn == "Archaeology Dating") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "archaeology_dating_simple"
    elif (scenarioNameIn == "Archaeology Dating") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "archaeology_dating_challenge"
    # Scenario 3: Plant Nutrients
    elif (scenarioNameIn == "Plant Nutrients") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "plant_nutrients_easy"
    elif (scenarioNameIn == "Plant Nutrients") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "plant_nutrients_normal"
    elif (scenarioNameIn == "Plant Nutrients") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "plant_nutrients_challenge"
    # Scenario 4: Reactor Lab
    elif (scenarioNameIn == "Reactor Lab") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "reactor_lab_easy"
    elif (scenarioNameIn == "Reactor Lab") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "reactor_lab_normal"
    elif (scenarioNameIn == "Reactor Lab") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "reactor_lab_challenge"
    # Scenario 5: Lost in Translation
    elif (scenarioNameIn == "Lost in Translation") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "lost_in_translation_distilled"
    elif (scenarioNameIn == "Lost in Translation") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "lost_in_translation_easy"
    elif (scenarioNameIn == "Lost in Translation") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "lost_in_translation_hard"
    # Scenario 6: Space Sick
    elif (scenarioNameIn == "Space Sick") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "space_sick_easy"
    elif (scenarioNameIn == "Space Sick") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "space_sick_normal"
    elif (scenarioNameIn == "Space Sick") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "space_sick_challenge"
    # Scenario 7: Proteomics
    elif (scenarioNameIn == "Proteomics") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):
        scenarioName = "proteomics_easy"
    elif (scenarioNameIn == "Proteomics") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "proteomics_normal"
    elif (scenarioNameIn == "Proteomics") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "proteomics_challenge"
    # Scenario 8: It's (not) Rocket Science!
    elif (scenarioNameIn == "It's (not) Rocket Science!") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["easy"]):      #### DEBUG?
        scenarioName = "not_rocket_science_easy"
    elif (scenarioNameIn == "It's (not) Rocket Science!") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "not_rocket_science_normal"
    elif (scenarioNameIn == "It's (not) Rocket Science!") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["challenge"]):
        scenarioName = "not_rocket_science_challenge"

    elif (scenarioNameIn == "Small Skills: Dialog Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_dialog_test"

    elif (scenarioNameIn == "Small Skills: Pick and Place Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_pickandplace_test"

    elif (scenarioNameIn == "Small Skills: Pick and Give Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_pickandgive_test"

    elif (scenarioNameIn == "Small Skills: Instrument Measurement Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_measurement_test"

    elif (scenarioNameIn == "Small Skills: Doors Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_doors_test"

    elif (scenarioNameIn == "Small Skills: Doors with Keys Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_doors_keys_test"

    elif (scenarioNameIn == "Small Skills: Navigation in a House Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_navigation_house_test"

    elif (scenarioNameIn == "Small Skills: Search Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_search_test"

    elif (scenarioNameIn == "Small Skills: Discovery Feed Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_discoveryfeed_test"

    elif (scenarioNameIn == "Small Skills: Moving Agents Test") and (difficulty == SCENARIO_DIFFICULTY_OPTIONS["normal"]):
        scenarioName = "smallskills_moving_agents_test"

    # Return the internal scenario name to use
    return scenarioName

class ScenarioMaker():
    # Constructor
    def __init__(self, world, seed:int=None):
        self.world = world
        self.world.randomSeed = seed
        self.world.rng = random.Random(seed)

    # Make a scenario
    def setupScenario(self, scenarioName:str, numUserAgents:int=1):
        if (scenarioName == "space_sick_normal"):
            scoringInfo = makeScenarioTown(self.world, numUserAgents)
            self.world.addTaskByName("SpaceSickTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "space_sick_easy"):
            scoringInfo = makeScenarioSpaceSickEasy(self.world, numUserAgents)
            self.world.addTaskByName("SpaceSickTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "space_sick_challenge"):
            scoringInfo = makeScenarioTownChallenge(self.world, numUserAgents)
            self.world.addTaskByName("SpaceSickTaskChallenge", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "combinatorial_chemistry_easy"):
            scoringInfo = makeScenarioStorageShedEasyDistilled(self.world, numUserAgents)
            self.world.addTaskByName("RustedKeyTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "combinatorial_chemistry_normal"):
            scoringInfo = makeScenarioStorageShed(self.world, numUserAgents)
            self.world.addTaskByName("RustedKeyTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "combinatorial_chemistry_challenge"):
            scoringInfo = makeScenarioStorageShedChallenge(self.world, numUserAgents)
            self.world.addTaskByName("RustedKeyTaskChallenge", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "archaeology_dating_easy"):
            scoringInfo = makeScenarioArchaeologicalDigEasyDistilled(self.world, numUserAgents)
            self.world.addTaskByName("ArchaeologyDigTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "archaeology_dating_simple"):
            scoringInfo = makeScenarioArchaeologicalDig(self.world, numUserAgents)
            self.world.addTaskByName("ArchaeologyDigTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "archaeology_dating_challenge"):
            scoringInfo = makeScenarioArchaeologicalDigGenericRadioisotope(self.world, numUserAgents)
            self.world.addTaskByName("ArchaeologyDigTaskGenericRadioisotope", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "plant_nutrients_easy"):
            scoringInfo = makeScenarioPlantGrowingEasy(self.world, numUserAgents)
            self.world.addTaskByName("SoilNutrientTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "plant_nutrients_normal"):
            scoringInfo = makeScenarioPlantGrowing(self.world, numUserAgents)
            self.world.addTaskByName("SoilNutrientTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "plant_nutrients_challenge"):
            scoringInfo = makeScenarioPlantGrowingChallenge(self.world, numUserAgents)
            self.world.addTaskByName("SoilNutrientTaskChallenge", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "lost_in_translation_distilled"):
            scoringInfo = makeScenarioRosettaStoneEasy(self.world, numUserAgents)
            self.world.addTaskByName("RosettaStoneTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName.startswith("lost_in_translation")):
            difficulty = scenarioName.rsplit("_")[-1]
            scoringInfo = makeScenarioRosettaStone(self.world, numUserAgents, difficulty)
            self.world.addTaskByName("RosettaStoneTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "reactor_lab_easy"):
            scoringInfo = makeScenarioReactorLabEasy(self.world, numUserAgents)
            self.world.addTaskByName("ReactorTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "reactor_lab_normal"):
            scoringInfo = makeScenarioReactorLab(self.world, numUserAgents)
            self.world.addTaskByName("ReactorTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "reactor_lab_challenge"):
            scoringInfo = makeScenarioReactorLabChallenge(self.world, numUserAgents)
            self.world.addTaskByName("ReactorTaskChallenge", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "proteomics_easy"):
            scoringInfo = makeScenarioProteomicsEasyDistilled(self.world, numUserAgents)
            self.world.addTaskByName("ProteomicsTaskEasy", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "proteomics_normal"):
            scoringInfo = makeScenarioProteomics(self.world, numUserAgents, challengeVersion=False)
            self.world.addTaskByName("ProteomicsTaskNormal", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "proteomics_challenge"):
            scoringInfo = makeScenarioProteomics(self.world, numUserAgents, challengeVersion=True)
            self.world.addTaskByName("ProteomicsTaskChallenge", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName.startswith("not_rocket_science")):
            difficulty = scenarioName.rsplit("_")[-1]
            scoringInfo = makeScenarioNotRocketScience(self.world, numUserAgents, difficulty)
            self.world.addTaskByName("NotRocketScienceTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName.startswith("tutorial")):
            scoringInfo = makeScenarioTutorial(self.world, numUserAgents)
            self.world.addTaskByName("TutorialTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_dialog_test"):
            scoringInfo = makeScenarioDialogTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsDialogTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_pickandplace_test"):
            scoringInfo = makeScenarioPickAndPlaceTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsPickAndPlaceTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_pickandgive_test"):
            scoringInfo = makeScenarioPickAndGiveTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsPickAndGiveTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_measurement_test"):
            scoringInfo = makeScenarioInstrumentMeasurementTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsInstrumentMeasurementTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_doors_test"):
            scoringInfo = makeScenarioDoorsTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsDoorsTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_doors_keys_test"):
            scoringInfo = makeScenarioDoorsKeysTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsDoorsKeysTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_navigation_house_test"):
            scoringInfo = makeScenarioNavigationHouseTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsNavigationHouseTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_search_test"):
            scoringInfo = makeScenarioSearchTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsSearchTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_discoveryfeed_test"):
            scoringInfo = makeScenarioDiscoveryFeedTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsDiscoveryFeedTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        elif (scenarioName == "smallskills_moving_agents_test"):
            scoringInfo = makeScenarioMovingAgentsTest(self.world, numUserAgents)
            self.world.addTaskByName("SmallSkillsMovingAgentsTask", scoringInfo)
            self.world.initialFilter()
            return (True, "")

        # If we reach here, the scenario was not recognized
        print("ERROR: setupScenario: scenarioName not recognized: " + scenarioName)
        return (False, "ERROR: setupScenario: scenarioName not recognized: " + scenarioName)
