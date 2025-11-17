from discoveryworld.scenarios.archaeology import makeScenarioArchaeologicalDig, makeScenarioArchaeologicalDigEasyDistilled
from discoveryworld.scenarios.archaeology import makeScenarioArchaeologicalDigGenericRadioisotope
from discoveryworld.scenarios.plant_growing import makeScenarioPlantGrowing, makeScenarioPlantGrowingEasy
from discoveryworld.scenarios.plant_growing_challenge import makeScenarioPlantGrowingChallenge
from discoveryworld.scenarios.rosetta_stone import makeScenarioRosettaStone
from discoveryworld.scenarios.rosetta_stone_easy import makeScenarioRosettaStoneEasy
from discoveryworld.scenarios.storage_shed import makeScenarioStorageShed, makeScenarioStorageShedChallenge, makeScenarioStorageShedEasyDistilled
from discoveryworld.scenarios.reactor_lab import makeScenarioReactorLab
from discoveryworld.scenarios.town import makeScenarioTown, makeScenarioTownChallenge
from discoveryworld.scenarios.space_sick_easy import makeScenarioSpaceSickEasy
from discoveryworld.scenarios.reactor_lab import makeScenarioReactorLab, makeScenarioReactorLabEasy, makeScenarioReactorLabChallenge
from discoveryworld.scenarios.proteomics import makeScenarioProteomics, makeScenarioProteomicsEasyDistilled
from discoveryworld.scenarios.not_rocket_science import makeScenarioNotRocketScience
from discoveryworld.scenarios.tutorial import makeScenarioTutorial
from discoveryworld.scenarios.smallskills_dialog import *
from discoveryworld.scenarios.smallskills_pickandplace import *
from discoveryworld.scenarios.smallskills_pickandgive import *
from discoveryworld.scenarios.smallskills_measurement import *
from discoveryworld.scenarios.smallskills_doors import *
from discoveryworld.scenarios.smallskills_doors_keys import *
from discoveryworld.scenarios.smallskills_navigation_house import *
from discoveryworld.scenarios.smallskills_search import *
from discoveryworld.scenarios.smallskills_discoveryfeed import *
from discoveryworld.scenarios.smallskills_moving_agents import *


SCENARIOS = {
    "Town": makeScenarioTown,
    "StorageShed": makeScenarioStorageShed,
    "PlantGrowing": makeScenarioPlantGrowing,
    "ArchaeologicalDig": makeScenarioArchaeologicalDig,
    "ArchaeologicalDigGenericRadioisotope": makeScenarioArchaeologicalDigGenericRadioisotope,
    "RosettaStone": makeScenarioRosettaStone,
    "Proteomics": makeScenarioProteomics,
    "NotRocketScience": makeScenarioNotRocketScience,
    "ReactorLab": makeScenarioReactorLab,
    "Tutorial": makeScenarioTutorial,
}
