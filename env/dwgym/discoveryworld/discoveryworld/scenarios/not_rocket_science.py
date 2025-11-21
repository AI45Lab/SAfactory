# reactor_lab.py

import random

import numpy as np
from termcolor import colored
from discoveryworld.ActionHistory import ActionType
from discoveryworld.Agent import NPC, Agent, NPCDevice
from discoveryworld.DialogTree import DialogMaker, DialogNode, DialogTree
from discoveryworld.Layer import Layer
from discoveryworld.TaskScorer import ScorecardElement, Task

from discoveryworld.buildings.terrain import mkPathX, mkPathY, mkSandFill, mkSignVillage

from discoveryworld.objects.Terrain import LaunchPad, Sand, SandPath


ROCKET_FORMULAS = """
   /\                 -= Rocketry Cheat Sheet =-
  /  \\
 /____\  Formulas:
 |    |    - Escape velocity: v = sqrt(2 * G * M / R)
 | D  |    - Orbital speed: O_v = sqrt(G * M / (R + h))
 | i  |    - Orbital period: O_T = 2 * pi * sqrt((R + h)^3 / (G * M))
 | s  |    - Gravitational acceleration: g = G * M / R^2
 | c  |    - Pendulum period: T = 2 * pi * sqrt(L / g)
 |----|    - Centripetal acceleration: a = v^2 / R
 | o  |    - Centripetal force: F = m * v^2 / R
 | v  |    - Thrust-to-weight ratio: TWR = F / (m * g)
 | e  |    - Specific impulse: Isp = F / (m_dot * g)
 | r  |    - Rocket equation: delta_v = Isp * g * ln(m0 / m1)
 | y  |
 |----|  Constants & variables:
 |  W |    - G: gravitational constant (6.67430e-11 N m^2 / kg^2)
 |  o |    - M: mass of planet (kg)
 |  r |    - R: radius of planet (m)
 |  l |    - h: height above planet (m)
 |  d |    - g: acceleration due to gravity (m/s^2)     _____
 |____|    - v: velocity (m/s)                      ,-:` \;',`'-,
/|    |\   - T: period (s)                        .'-;_,;  ':-;_,'.
 ||  ||    - L: length of pendulum (m)           /;   '/    ,  _`.-\\
 ||  ||    - F: force (N)                       | '`. (`     /` ` \`|
 ||  ||    - a: acceleration (m/s^2)            |:.  `\`-.   \_   / |
/_|__|_\   - m_dot: mass flow rate (kg/s)       |     (   `,  .`\ ;'|
 ||||||    - m0: initial mass of rocket (kg)     \     | .'     `-'/
 /||||\    - m1: final mass of rocket (kg)        `.   ;/        .'
  /||\     - delta_v: change in velocity (m/s)      `'-.______.-'
"""


PLANETS = [
    "Earth",
    "the Moon",
    "Mars",
    "Io",
    "Europa",
    "Mercury",
    "Venus",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
    "Ganymede",
]

# According to https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
PLANET_RADII = {
    "Mercury": 2439.4,
    "Venus": 6052,
    "Earth": 6371,
    "Mars": 3389.5,
    "Jupiter": 69911,
    "Saturn": 58232,
    "Uranus": 25362,
    "Neptune": 24622,
    "Pluto": 1188.3,
    "the Moon": 1737.5,
    "Io": 1821.6,
    "Europa": 1560.8,
    "Ganymede": 2634.1,
}

PLANET_MASSES = {
    "Mercury": 330.11e21,
    "Venus": 4867.4e21,
    "Earth": 5972.4e21,
    "Mars": 641.71e21,
    "Jupiter": 1898187e21,
    "Saturn": 568317e21,
    "Uranus": 86813e21,
    "Neptune": 102413e21,
    "Pluto": 13.03e21,
    "the Moon": 73.46e21,
    "Io": 89.32e21,
    "Europa": 48.00e21,
    "Ganymede": 148.2e21,
}


def mkControlRoom(world, x, y):
    layout = [
        "#######",
        "#p#m  #",
        "# #tTt#",
        "#     #",
        "####ds#",
    ]
    bounds = (x, y, x+len(layout[0]), y+len(layout))

    pendulum = world.createObject("Pendulum")
    launchTerminal = LaunchTerminal(world) #world.createObject("LaunchTerminal")
    for j, row in enumerate(layout):
        for i, char in enumerate(row):
            if char == "#":
                world.addObject(x+i, y+j, Layer.BUILDING, world.createObject("Wall"))
            elif char == ".":
                pass
            else:
                world.addObject(x+i, y+j, Layer.BUILDING, world.createObject("Floor"))

            if char == "d": # Add door and floor
                door = world.createObject("Door")
                world.addObject(x+i, y+j, Layer.FURNITURE, door)
            elif char == "s":  # Add wall and sign
                world.addObject(x+i, y+j, Layer.BUILDING, world.createObject("Wall"))
                sign = world.createObject("Sign", variant=1)
                sign.setText("Control Room")
                world.addObject(x+i, y+j, Layer.FURNITURE, sign)
            elif char == "T":  # Add table and chairs and computer
                table = world.createObject("Table")
                world.addObject(x+i, y+j, Layer.FURNITURE, table)
                table.addObject(launchTerminal)
            elif char == "t":  # Add table and chairs
                world.addObject(x+i, y+j, Layer.FURNITURE, world.createObject("Table"))
            elif char == "m":  # Add monitor
                world.addObject(x+i, y+j, Layer.OBJECTS, world.createObject("LaunchMonitor", part="bl"))
                world.addObject(x+i+1, y+j, Layer.OBJECTS, world.createObject("LaunchMonitor", part="b"))
                world.addObject(x+i+2, y+j, Layer.OBJECTS, world.createObject("LaunchMonitor", part="br"))
                world.addObject(x+i, y+j-1, Layer.OBJECTS, world.createObject("LaunchMonitor", part="tl"))
                world.addObject(x+i+1, y+j-1, Layer.OBJECTS, world.createObject("LaunchMonitor", part="t"))
                world.addObject(x+i+2, y+j-1, Layer.OBJECTS, world.createObject("LaunchMonitor", part="tr"))
            elif char == "p":  # Add pendulum
                world.addObject(x+i, y+j, Layer.OBJECTS, pendulum)

    return pendulum, launchTerminal, bounds


def mkThrustTestingGround(world, x, y):
    layout = [
        "sABC   T",
        "########",
    ]

    loadCell = LoadCellInterface(world)
    rocket_top = world.createObject("Rocket", part="flip_top")
    rocket_bottom = world.createObject("Rocket", part="flip_bottom")
    loadCell.rocket = (rocket_top, rocket_bottom)
    #rocket_top = world.createObject("Rocket", part="flip_fire")

    for j, row in enumerate(layout):
        for i, char in enumerate(row):
            if char in ("A", "B", "C"):
                world.addObject(x+i, y+j-2, Layer.AIR, world.createObject("FuelTank", variant=char, part="t"))
                world.addObject(x+i, y+j-1, Layer.AIR, world.createObject("FuelTank", variant=char, part="c"))
                world.addObject(x+i, y+j, Layer.FURNITURE, world.createObject("FuelTank", variant=char, part="b"))

            elif char == "s":
                sign = world.createObject("Sign")
                sign.setText("Propellant Tanks")
                world.addObject(x+i, y+j, Layer.OBJECTS, sign)

            elif char == "T":
                world.addObject(x+i-2, y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="l"))
                world.addObject(x+i-1, y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="c"))
                world.addObject(x+i  , y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="r"))

                world.addObject(x+i-2, y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="0,5", variant="top"))
                world.addObject(x+i-1, y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="1,5", variant="top"))
                world.addObject(x+i-0, y+j-3, Layer.AIR, world.createObject("LoadCellWall", part="2,5", variant="top"))
                world.addObject(x+i-2, y+j-2, Layer.AIR, world.createObject("LoadCellWall", part="0,6", variant="top", isPassable=False))
                world.addObject(x+i-0, y+j-2, Layer.AIR, world.createObject("LoadCellWall", part="2,6", variant="top", isPassable=False))

                world.addObject(x+i-2, y+j-2, Layer.FURNITURE, world.createObject("LoadCellWall", part="bl", isPassable=False))
                world.addObject(x+i-1, y+j-2, Layer.FURNITURE, world.createObject("LoadCellWall", part="b", isPassable=False))
                world.addObject(x+i  , y+j-2, Layer.FURNITURE, world.createObject("LoadCellWall", part="br", isPassable=False))

                world.addObject(x+i-2, y+j-0, Layer.FURNITURE, world.createObject("LoadCellWall", part="bl", isPassable=False))
                world.addObject(x+i  , y+j-0, Layer.FURNITURE, world.createObject("LoadCellWall", part="br", isPassable=False))

                world.addObject(x+i-2, y+j-2, Layer.WORLD, world.createObject("LaunchPad", variant="cc"))
                world.addObject(x+i-0, y+j-2, Layer.WORLD, world.createObject("LaunchPad", variant="cc"))
                world.addObject(x+i-1, y+j-2, Layer.WORLD, world.createObject("LaunchPad", variant="cc"))
                world.addObject(x+i-2, y+j-1, Layer.FURNITURE, world.createObject("LaunchPad", variant="cc"))
                world.addObject(x+i-1, y+j-1, Layer.FURNITURE, world.createObject("LaunchPad", variant="cc"))
                world.addObject(x+i-0, y+j-1, Layer.FURNITURE, world.createObject("LaunchPad", variant="cc"))

                # world.addObject(x+i-1, y+j-3, Layer.AIR, world.createObject("Rocket", part="flip_fire"))
                world.addObject(x+i-1, y+j-2, Layer.FURNITURE, rocket_bottom)
                world.addObject(x+i-1, y+j-1, Layer.FURNITURE, rocket_top)

                # world.addObject(x+i-1, y+j-3, Layer.AIR, world.createObject("Rocket", part="flip_fire"))
                # world.addObject(x+i-1, y+j-2, Layer.FURNITURE, world.createObject("Rocket", part="flip_bottom_fire"))
                # world.addObject(x+i-1, y+j-1, Layer.FURNITURE, world.createObject("Rocket", part="flip_top_fire"))

                world.addObject(x+i-2, y+j-1, Layer.AIR, world.createObject("LoadCell", part="tl"))
                world.addObject(x+i-1, y+j-1, Layer.AIR, world.createObject("LoadCell", part="t"))
                world.addObject(x+i, y+j-1, Layer.AIR, world.createObject("LoadCell", part="tr"))
                world.addObject(x+i-2, y+j, Layer.FURNITURE, world.createObject("LoadCell", part="bl"))
                world.addObject(x+i-1, y+j, Layer.FURNITURE, loadCell)
                world.addObject(x+i, y+j, Layer.FURNITURE, world.createObject("LoadCell", part="br"))

                world.addObject(x+i-2, y+j-1, Layer.AIR, world.createObject("LoadCellWall", part="0,7", variant="top", isPassable=False))
                world.addObject(x+i-0, y+j-1, Layer.AIR, world.createObject("LoadCellWall", part="2,7", variant="top", isPassable=False))

    return loadCell


def mkRocket(world, x, y):
    world.addObject(x, y, Layer.OBJECTS, world.createObject("Rocket", part="bottom"))
    world.addObject(x, y-1, Layer.AIR, world.createObject("Rocket", part="top"))


def mkLaunchPad(world, x, y, width, height):
    # Check that it has a minimum size.
    if width < 3 or height < 3:
        print(f"Error: Launch pad is too small: {width} x {height}")
        return

    # Top-left corner
    world.addObject(x, y, Layer.BUILDING, world.createObject("LaunchPad", variant="tl"))
    # Top-right corner
    world.addObject(x + width - 1, y, Layer.BUILDING, world.createObject("LaunchPad", variant="tr"))
    # Bottom-left corner
    world.addObject(x, y + height - 1, Layer.BUILDING, world.createObject("LaunchPad", variant="bl"))
    # Bottom-right corner
    world.addObject(x + width - 1, y + height - 1, Layer.BUILDING, world.createObject("LaunchPad", variant="br"))
    # Top border
    for i in range(1, width - 1):
        world.addObject(x + i, y, Layer.BUILDING, world.createObject("LaunchPad", variant="tc"))

    # Bottom wall
    for i in range(1, width - 1):
        world.addObject(x + i, y + height - 1, Layer.BUILDING, world.createObject("LaunchPad", variant="bc"))

    # Left wall
    for i in range(1, height - 1):
        world.addObject(x, y + i, Layer.BUILDING, world.createObject("LaunchPad", variant="lc"))

    # Right wall
    for i in range(1, height - 1):
        world.addObject(x + width - 1, y + i, Layer.BUILDING, world.createObject("LaunchPad", variant="rc"))

    # Floor
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            world.addObject(x + i, y + j, Layer.BUILDING, world.createObject("LaunchPad", variant="cc"))

    bounds = (x+1, y+1, x + width - 2, y + height - 2)
    return bounds



def makeScenarioNotRocketScience(world, numUserAgents=1, difficulty="easy"):
    scoringInfo = {}
    scoringInfo["criticalHypotheses"] = []
    scoringInfo["criticalQuestions"] = []

    scoringInfo["difficulty"] = difficulty

    planet = "X"
    if difficulty == "easy":
        planet = PLANETS[world.randomSeed % len(PLANETS)]
        scoringInfo["criticalHypotheses"].append(f"The planet's caracteristics are very similar to {planet}.")

    scoringInfo["planet"] = planet

    # Critical values for the scenario.
    # Planet radius
    planetRadius = world.rng.randint(400, 10000)  # (km)
    if difficulty == "easy":
        planetRadius = PLANET_RADII[planet]  # (km)

    worldHeight = (world.sizeY-1) / 1000  # (km)
    diffAngle = (360 * worldHeight) / (2*planetRadius*np.pi)  # (degrees)
    startAngle = world.rng.randint(2, 5) + (world.rng.random() - 0.5) * diffAngle * 15  # (degrees)
    ##print(f"DiffAngle: {diffAngle} {startAngle}")

    scoringInfo["planetRadius"] = planetRadius
    scoringInfo["criticalHypotheses"].append(f"Planet radius is {planetRadius} km.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the planet's radius is approximately " + str(planetRadius) + " km?")

    # Planet mass
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # (N m^2 / kg^2)
    MIN_PLANET_MASS = 1e23  # kg
    MAX_PLANET_MASS = 1e25  # kg
    planetMass = MIN_PLANET_MASS + world.rng.random() * (MAX_PLANET_MASS - MIN_PLANET_MASS)  # (kg)
    if difficulty == "easy":
        planetMass = PLANET_MASSES[planet]  # (kg)

    scoringInfo["planetMass"] = planetMass
    scoringInfo["criticalHypotheses"].append(f"Planet mass is {planetMass} kg.")

    # Planet gravity
    planetGravity = GRAVITATIONAL_CONSTANT * planetMass / (planetRadius * 1000) ** 2  # (m/s^2)
    scoringInfo["planetGravity"] = planetGravity
    scoringInfo["criticalHypotheses"].append(f"Planet gravity is {planetGravity} m/s^2.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the planet's gravity is approximately " + str(round(planetGravity, 2)) + " m/s^2?")

    # 1-meter pendulum period
    pendulumLength = 1  # (m)
    scoringInfo["pendulumLength"] = pendulumLength
    scoringInfo["criticalHypotheses"].append(f"The pendulum's length is {pendulumLength}m.")
    pendulumPeriod = 2 * np.pi * np.sqrt(pendulumLength / planetGravity)  # (s)
    scoringInfo["pendulumPeriod"] = pendulumPeriod
    scoringInfo["criticalHypotheses"].append(f"The pendulum's period is {pendulumPeriod} seconds.")
    if (difficulty != "easy"):
        scoringInfo["criticalQuestions"].append("Does it clearly state the pendulum's period is approximately " + str(round(pendulumPeriod, 2)) + " seconds?")

    # Target orbit height
    orbitHeight = world.rng.randint(100, 10000)  # (km)
    if difficulty == "easy":
        orbitHeight = 400  # e.g. ISS (km)

    scoringInfo["orbitHeight"] = orbitHeight
    scoringInfo["criticalHypotheses"].append(f"Target orbit height is {scoringInfo['orbitHeight']} km.")

    # Target orbital speed
    orbitSpeed = np.sqrt(GRAVITATIONAL_CONSTANT * planetMass / ((planetRadius + scoringInfo["orbitHeight"])*1000))  # m/s
    scoringInfo["orbitSpeed"] = orbitSpeed
    scoringInfo["criticalHypotheses"].append(f"Target orbit speed is {scoringInfo['orbitSpeed']} m/s.")
    scoringInfo["criticalQuestions"].append("Does it clearly state the target orbit speed to reach an orbital height of " + str(scoringInfo['orbitHeight']) + " km is approximately " + str(round(orbitSpeed, 1)) + " m/s? (plus or minus 3 m/s)")

    # Rocket mass
    MIN_ROCKET_MASS = 1e3  # kg
    MAX_ROCKET_MASS = 1e4  # kg
    rocketMass = float(world.rng.randint(MIN_ROCKET_MASS, MAX_ROCKET_MASS))  # kg
    scoringInfo["criticalHypotheses"].append(f"The dry mass of the rocket is {rocketMass} kg.")

    rocketTanksVolume = 10000  # L
    scoringInfo["criticalHypotheses"].append(f"The rocket's tanks have a volume of {rocketTanksVolume}L.")

    def sampleFuel(rng, desiredFuel):
        fuelDensity = np.logspace(-2.5, 2, 70)
        massFlowRate = 1 + np.linspace(0, 1000, 200) * 1e1  # kg/s
        thrust = 1 + np.linspace(0, 1500, 200)*1e4  # N

        massFlowRate, thrust, fuelDensity = np.meshgrid(massFlowRate, thrust, fuelDensity)

        specificImpulse = thrust / (massFlowRate * planetGravity)  # s
        effectiveExhaustVelocity = specificImpulse * planetGravity  # m/s

        deltaV = orbitSpeed  # m/s
        rocketWithFuelMass = rocketMass * np.exp(deltaV / effectiveExhaustVelocity)  # kg
        fuelMassNeeded = rocketWithFuelMass - rocketMass  # kg
        fuelAmount = fuelMassNeeded / fuelDensity  # L

        thrustWeightRatio = thrust / (rocketWithFuelMass * planetGravity)

        fuelAmountOkay = fuelAmount <= rocketTanksVolume
        thrustWeightRatioAbove1 = thrustWeightRatio > 1

        if desiredFuel == "Adequate":
            mask = fuelAmountOkay & thrustWeightRatioAbove1
        elif desiredFuel == "Not enough thrust":
            mask = fuelAmountOkay & ~thrustWeightRatioAbove1
        elif desiredFuel == "Too much fuel":
            mask = ~fuelAmountOkay & thrustWeightRatioAbove1
        else:
            NotImplementedError("Desired fuel type not implemented")

        assert np.any(mask), f"Could not find a fuel that satisfies the desired properties."

        # Sample a fuel that satisfies the desired properties
        idx = rng.choice(np.where(mask.ravel())[0])
        fuel = {
            "density": fuelDensity.ravel()[idx],  # kg/L
            "massFlowRate": massFlowRate.ravel()[idx],  # kg/s
            "thrust": thrust.ravel()[idx],  # N
        }
        return fuel

    # Because we assume a fix engine, thrust and mass flow rate will be determined by the fuel type.
    fuels = [sampleFuel(world.rng, desiredFuel) for desiredFuel in ["Adequate", "Not enough thrust", "Too much fuel"]]

    # Select a random fuel type
    fuelTypes = ["A", "B", "C"]
    world.rng.shuffle(fuelTypes)

    def _computeStatsForFuel(fuel, fuelType):
        fuelDensity = fuel["density"]  # kg/L
        massFlowRate = fuel["massFlowRate"]  # kg/s
        scoringInfo["criticalHypotheses"].append(f"Fuel {fuelType}'s density is {fuelDensity} kg/L.")
        scoringInfo["criticalHypotheses"].append(f"The mass flow rate for fuel {fuelType} is {massFlowRate} kg/s.")

        thrust = fuel["thrust"]  # N
        scoringInfo["criticalHypotheses"].append(f"The thrust generated by fuel {fuelType} is {thrust/1000} kN.")

        specificImpulse = thrust / (massFlowRate * planetGravity)  # s
        effectiveExhaustVelocity = specificImpulse * planetGravity  # m/s
        scoringInfo["criticalHypotheses"].append(f"The specific impulse with fuel {fuelType} is {specificImpulse} s and its effective exhaust velocity is {effectiveExhaustVelocity} m/s.")

        deltaV = scoringInfo["orbitSpeed"]  # m/s
        rocketWithFuelMass = rocketMass * np.exp(deltaV / effectiveExhaustVelocity)  # kg
        fuelMassNeeded = rocketWithFuelMass - rocketMass  # kg
        fuelAmount = fuelMassNeeded / fuelDensity  # L
        scoringInfo["criticalHypotheses"].append(f"The total mass of the rocket at launch with fuel {fuelType} would be {rocketWithFuelMass} kg ({fuelMassNeeded/rocketWithFuelMass:%} is fuel).")

        scoringInfo["criticalHypotheses"].append(f"The amount of fuel {fuelType} needed would be {fuelAmount}L ({fuelAmount/rocketTanksVolume:%} of rocket's tanks).")
        if fuelAmount > rocketTanksVolume:
            scoringInfo["criticalHypotheses"].append(f"That won't fit in the rocket's tanks.")

        thrustWeightRatio = thrust / (rocketWithFuelMass * planetGravity)
        scoringInfo["criticalHypotheses"].append(f"Fuel {fuelType} would produce a thrust-to-weight ratio of {thrustWeightRatio} times the planet's gravity.")

        if thrustWeightRatio <= 1:
            scoringInfo["criticalHypotheses"][-1] += f"That's not enough for liftoff!"

        return fuelAmount, thrustWeightRatio

    fuelAmount, thrustWeightRatio = _computeStatsForFuel(fuels[2], fuelTypes[2])
    assert fuelAmount > rocketTanksVolume, f"Expect this distractor fuel to be not dense enough ({fuelAmount} vs. {rocketTanksVolume})"
    assert thrustWeightRatio > 1, f"Expect this distractor fuel to have enough thrust ({thrustWeightRatio}g)"

    fuelAmount, thrustWeightRatio = _computeStatsForFuel(fuels[1], fuelTypes[1])
    assert fuelAmount <= rocketTanksVolume, f"Expect this distractor fuel to be dense enough ({fuelAmount} vs. {rocketTanksVolume})"
    assert thrustWeightRatio <= 1, f"Expect this distractor fuel to not have enough thrust ({thrustWeightRatio}g)"

    fuelAmount, thrustWeightRatio = _computeStatsForFuel(fuels[0], fuelTypes[0])
    scoringInfo["fuelAmount"] = fuelAmount

    scoringInfo["fuelType"] = fuelTypes[0]
    scoringInfo["criticalHypotheses"].append(f"The best fuel type is {fuelTypes[0]}.")

    assert thrustWeightRatio > 1, f"The thrust-to-weight ratio is too low, the rocket won't liftoff {thrustWeightRatio}g."
    assert fuelAmount <= rocketTanksVolume, f"The amount of fuel needed doesn't fit in the tanks. {fuelAmount} vs. {rocketTanksVolume}"

    if (difficulty == "challenge"):
        scoringInfo["criticalQuestions"].append(f"Does it clearly eliminate fuel type {fuelTypes[1]} as it wouldn't generate enough thrust to liftoff the rocket (i.e., its thrust to weight ratio is lower or equal to 1 g)?")
        scoringInfo["criticalQuestions"].append(f"Does it clearly eliminate fuel type {fuelTypes[2]} as its density is to low and would require more fuel than what the rocket can hold?")
        scoringInfo["criticalQuestions"].append(f"Does it clearly state what fuel type to use {scoringInfo['fuelType']} (between {','.join(sorted(fuelTypes))})?")
        scoringInfo["criticalQuestions"].append(f"Does it clearly state the amount needed of fuel type {scoringInfo['fuelType']} is approximately {int(scoringInfo['fuelAmount'])} liters? (plus or minus 2 liters)")

    # Set a limit for the number of user agents
    MAX_NUM_AGENTS = 3
    if (numUserAgents > MAX_NUM_AGENTS):
        numUserAgents = MAX_NUM_AGENTS

    # Populate with structures/objects

    # Fill with sand
    mkSandFill(world)

    # Buildings
    # Launchpad
    mkLaunchPad(world, 13, 0, 7, 5)
    mkRocket(world, 16, 2)

    # Rocket Control Operations
    pendulum, launchTerminal, controlRoomBounds = mkControlRoom(world, 7, 20)
    pendulum.setLength(scoringInfo["pendulumLength"])
    pendulum.oscillationPeriod = scoringInfo["pendulumPeriod"]
    scoringInfo["pendulum"] = pendulum
    launchTerminal.attributes["orbitHeight"] = scoringInfo["orbitHeight"]
    launchTerminal.targetOrbitSpeed = scoringInfo["orbitSpeed"]
    scoringInfo["launchTerminal"] = launchTerminal

    # Load cell test ground
    loadCell = mkThrustTestingGround(world, 22, 24)
    loadCell.fuels = {fuelType: fuel for fuelType, fuel in zip(fuelTypes, fuels)}
    loadCell.rocketDryMass = rocketMass
    loadCell.attributes["weight"] = rocketMass
    loadCell.rocketTanksVolume = 5000  # L
    scoringInfo["loadCell"] = loadCell

    launchTerminal.targetFuelType = scoringInfo["fuelType"]
    launchTerminal.targetFuelAmount = scoringInfo["fuelAmount"]

    if difficulty != "challenge":
        # Preset the fuel type and amount.
        launchTerminal.attributes["fuel"] = scoringInfo["fuelType"]
        launchTerminal.attributes["fuelAmount"] = int(scoringInfo["fuelAmount"])

    # Paths
    mkPathX(9, 25, 6, world, type="SandPath")
    mkPathX(18, 25, 13, world, type="SandPath")
    mkPathY(15, 4, 28, world, type="SandPath")
    mkPathY(16, 4, 28, world, type="SandPath")
    mkPathY(17, 4, 28, world, type="SandPath")

    # Add big village sign
    mkSignVillage(15, 27, world)

    # Add some plants
    plantCount = 0
    minPlants = 7
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(0, world.sizeX - 1)
        randY = world.rng.randint(0, world.sizeY - 1)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY)
        # Get types of objects
        objTypes = set(obj.type for obj in objs)

        # Check to see that there is only sand here
        if objTypes == {"sand"}:
            world.addObject(randX, randY, Layer.OBJECTS, world.createObject("Cactus", part="bottom"))
            world.addObject(randX, randY-1, Layer.AIR, world.createObject("Cactus", part="top"))

            plantCount += 1

    plantCount = 0
    minPlants = 7
    while (plantCount < minPlants):
        # Pick a random location
        randX = world.rng.randint(1, world.sizeX - 2)
        randY = world.rng.randint(1, world.sizeY - 2)

        # Check to see if there are any objects other than grass there
        objs = world.getObjectsAt(randX, randY) + world.getObjectsAt(randX+1, randY)

        # Get types of objects
        objTypes = set(obj.type for obj in objs)

        # Check to see that there is only sand here
        if objTypes == {"sand"}:
            world.addObject(randX, randY, Layer.OBJECTS, world.createObject("BigCactus", side="left", part="bottom"))
            world.addObject(randX, randY-1, Layer.AIR, world.createObject("BigCactus", side="left", part="top"))
            world.addObject(randX+1, randY, Layer.OBJECTS, world.createObject("BigCactus", side="right", part="bottom"))
            world.addObject(randX+1, randY-1, Layer.AIR, world.createObject("BigCactus", side="right", part="top"))

            plantCount += 1

    # Assign light angle to each Sand tile. # TODO: also to path and launchpad
    for y in range(world.sizeY):
        lightAngle = np.round((startAngle + diffAngle * (y / (world.sizeY-1))) * 1e3, 5)  # (millidegrees)
        # print(f"Light angle at row {y} is {lightAngle} degrees.")

        for x in range(world.sizeX):
            if x >= controlRoomBounds[0] and x < controlRoomBounds[2] and y >= controlRoomBounds[1] and y < controlRoomBounds[3]:
                continue

            for obj in world.getObjectsAt(x, y):
                # Check if the object is Sand.
                if isinstance(obj, (Sand, SandPath, LaunchPad)):
                    obj.attributes["lightAngle"] = lightAngle

    # Add dialog to objects.
    mkDialogLaunchTerminal(launchTerminal, difficulty)
    mkDialogLoadCellInterface(loadCell)

    # Add some number of user agents
    scoringInfo["speedSquare"] = []
    scoringInfo["rocketryBook"] = []
    for userAgentIdx in range(0, numUserAgents):
        userAgent = Agent(world)
        # Create tools
        rocketryBook = world.createObject("RocketryBook")
        rocketryBook.attributes["document"] = ROCKET_FORMULAS
        userAgent.addObject(rocketryBook)
        scoringInfo["rocketryBook"].append(rocketryBook)

        speedSquare = world.createObject("SpeedSquare")
        userAgent.addObject(speedSquare)
        scoringInfo["speedSquare"].append(speedSquare)

        # Add the agent to a specfic location
        world.addObject(16+userAgentIdx, 3, Layer.AGENT, userAgent)    # Near rocket
        # world.addObject(11+userAgentIdx, 23, Layer.AGENT, userAgent)   # In control room
        # world.addObject(23+userAgentIdx, 25, Layer.AGENT, userAgent)   # Near load cell
        # Register the agent with the World so we can keep track of it
        world.addAgent(userAgent)

    # Add teleport locations to world
    world.addTeleportLocation("control room", 11, 23)
    world.addTeleportLocation("rocket", 16, 3)
    world.addTeleportLocation("northern observation post", 6, 1)  # TODO: make sure there's not cactus here
    world.addTeleportLocation("southern observation post", 6, 31)  # TODO: make sure there's not cactus here
    world.addTeleportLocation("thrust test site", 23, 25)

    # Add signs at observation locations
    signNorth = world.createObject("Sign")
    signNorth.setText("This sign is exactly 30 meters north of the southern sign.")
    world.addObject(5, 1, Layer.OBJECTS, signNorth)
    signSouth = world.createObject("Sign")
    signSouth.setText("This sign is exactly 30 meters south of the northern sign.")
    world.addObject(5, 31, Layer.OBJECTS, signSouth)


    # Compute expected approximations for the scenario made the players.
    worldHeight = ((world.sizeY-1) - 0) #/ 1000  # (km)
    sandTileAt00 = [obj for obj in world.getObjectsAt(0, 0) if obj.type == "sand"][0]
    sandTileAt0N = [obj for obj in world.getObjectsAt(0, world.sizeY-1) if obj.type == "sand"][0]
    approxDiffLightAngle = abs(sandTileAt0N.attributes["lightAngle"] - sandTileAt00.attributes["lightAngle"])  # (millidegrees)
    approxDiffLightAngle = approxDiffLightAngle / 1e3  # (degrees)

    approxPlanetRadius = (360*worldHeight) / (approxDiffLightAngle * 2 * np.pi)  # (m)
    approxPlanetRadius = approxPlanetRadius / 1e3  # (km)

    #print(sandTileAt00.attributes["lightAngle"] , sandTileAt0N.attributes["lightAngle"] )
    #print(f"{diffAngle} vs. {approxDiffLightAngle}")
    ##print(f"Approximation error for DiffAngle: {abs(approxDiffLightAngle - diffAngle)}")

    ##print(f"{planetRadius} vs. {approxPlanetRadius}")
    ##print(f"Approximation error for PlanetRadius: {abs(approxPlanetRadius - planetRadius)}")

    # 1-meter pendulum period
    nbTicks = 50
    # nbOscillations = ticks // pendulumPeriod
    nbOscillations = np.round(nbTicks / pendulumPeriod, 3)  # TODO: needs to match the code in the Pendulum object

    # approximage the period of a 1-meter pendulum
    approxPendulumPeriod = nbTicks / nbOscillations

    # approximate g from the period of a 1-meter pendulum
    g = 4 * np.pi ** 2 * pendulumLength / pendulumPeriod ** 2
    approxG = 4 * np.pi ** 2 * pendulumLength / approxPendulumPeriod ** 2

    ##print(f"Approximation error for PendulumPeriod: {abs(approxPendulumPeriod - pendulumPeriod)}")
    ##print(f"Approximation error for g with pendulum: {abs(approxG - g)}")
    ##print(f"Approximation error for g: {abs(approxG - planetGravity)}")

    approxOrbitSpeed = np.sqrt((approxG * (approxPlanetRadius*1e3)**2) / ((approxPlanetRadius + orbitHeight)*1e3))  # m/s
    ##print(f"Approximation error for OrbitSpeed: {abs(approxOrbitSpeed - orbitSpeed)}")
    ##print(f"{orbitSpeed} vs. {approxOrbitSpeed}")

    assert abs(approxOrbitSpeed - orbitSpeed) < 1, f"Approximation error for OrbitSpeed: {abs(approxOrbitSpeed - orbitSpeed)}"
    assert orbitSpeed >= 0, f"OrbitSpeed is {orbitSpeed}"
    assert orbitSpeed <= 25000, f"OrbitSpeed is {orbitSpeed}"

    # Return scoring info
    return scoringInfo


class NotRocketScienceTask(Task):

    def __init__(self, world, scoringInfo):
        orbitHeight = scoringInfo["orbitHeight"]
        taskDescription = "You are at the new Launch Site on Planet X. "
        taskDescription += f"To better monitor the weather patterns, you were tasked to send a new probe on orbit at {orbitHeight} kilometers from the ground. "
        taskDescription += "You were told to simply punch in the target orbital speed needed for such an altitude in the launch terminal and the system will take care of the rest. "
        taskDescription += "There is only one single-stage rocket, and this is a very important mission, so we better get it right the first time! "

        if scoringInfo["difficulty"] == "challenge":
            taskDescription += "Upon arriving at the launch site, you realized the rocket hasn't been fueled yet. "
            taskDescription += "You will need to figure which one of the three types of propellant to use and how much to put in the rocket. "
            taskDescription += "The rocket can hold a maximum of 10,000 liters of fuel, and it's weight can be determined using the load cell. "

        if scoringInfo["difficulty"] == "easy":
            taskDescription += f"Thankfully, you know the planet's characteristics, such as its mass and radius, are exactly the same as {scoringInfo['planet']}. "

        taskDescription += "\nGood thing, you brought your faithful rocketry book with you!\n"
        taskDescription += "Here are some additional helpful notes: \n"
        taskDescription += "  1. The grid cells on Planet X are 1-meter square.\n"
        taskDescription += "  2. You can assume air resistance is neglible.\n"
        taskDescription += "  3. Pay attention to the units in the formulas.\n"
        taskDescription += "  4. The formulas are simplifications and may not be 100% accurate.\n"
        taskDescription += "  5. Keep as much decimal places in your calculations.\n"
        if (scoringInfo["difficulty"] != "easy"):
            taskDescription += "  6. Look around for two observation posts to the north and south.\n"

        taskDescription = taskDescription.strip()

        super().__init__("NotRocketScienceTask", taskDescription, world, scoringInfo)
        self.score = 0
        self.maxScore = 0

        # Scorecard elements

        self.scorecardReadRocketryBook = ScorecardElement("Read the rocketry book", "The agent has read the rocketry book.", maxScore=1)
        self.scoreCard.append(self.scorecardReadRocketryBook)

        self.scorecardUseLaunchTerminal = ScorecardElement("Use launch terminal", "The agent has seen the launch terminal interface.", maxScore=1)
        self.scoreCard.append(self.scorecardUseLaunchTerminal)

        self.scorecardEnterOrbitVelocity = ScorecardElement("Enter correct orbit velocity", "The agent has entered the correct target orbital velocity.", maxScore=1)
        self.scoreCard.append(self.scorecardEnterOrbitVelocity)

        if scoringInfo["difficulty"] != "easy":
            self.scorecardMeasureLightAngle = ScorecardElement("Measure light angle", "The agent has measured light angle at two locations.", maxScore=2)
            self.scoreCard.append(self.scorecardMeasureLightAngle)

            self.scorecardActivatePendulum = ScorecardElement("Activate pendulum", "Pendulum was activated.", maxScore=1)
            self.scoreCard.append(self.scorecardActivatePendulum)

            self.scorecardReadoutPendulum = ScorecardElement("Observe pendulum", "The agent has read the pendulum.", maxScore=1)
            self.scoreCard.append(self.scorecardReadoutPendulum)

        if scoringInfo["difficulty"] == "challenge":
            self.scorecardUseLoadCellInterface = ScorecardElement("Use load cell interface", "The agent has seen the load cell interface.", maxScore=1)
            self.scoreCard.append(self.scorecardUseLoadCellInterface)
            self.scorecardTryDifferentFuel = ScorecardElement("Try the different fuel types", "The agent has tested the different fuel types with the load cell.", maxScore=3)
            self.scoreCard.append(self.scorecardTryDifferentFuel)
            self.scorecardEnterFuelAndAmount = ScorecardElement("Enter correct fuel type and amount", "The agent has entered the correct fuel type and its amount.", maxScore=1)
            self.scoreCard.append(self.scorecardEnterFuelAndAmount)

        # Add hypotheses from scoringInfo
        self.criticalHypotheses = scoringInfo["criticalHypotheses"]
        self.criticalQuestions = scoringInfo["criticalQuestions"]

        # Update max score based on the scorecard elements.
        self.maxScore = sum(element.maxScore for element in self.scoreCard)

    # Update the task progress
    def updateTick(self):
        if self.completed:
            return  # Do not update the score if the task is already marked as completed

        # Check if player has read the rocketry book.
        if not self.scorecardReadRocketryBook.completed:
            rocketryBook = self.scoringInfo["rocketryBook"][0]  # TODO: deal with multiple rocketry books
            for agent in self.world.agents:
                if agent.actionHistory.queryActionObjects(ActionType.READ, arg1=rocketryBook, stopAtFirst=True):
                    self.scorecardReadRocketryBook.updateScore(1, True, associatedUUIDs=[agent.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has read the rocketry book (UUID: {rocketryBook.uuid}).")

        launchTerminal = self.scoringInfo["launchTerminal"]
        # Check if player has used the launch terminal.
        if not self.scorecardUseLaunchTerminal.completed:
            if "interactedWith" in launchTerminal.attributes["states"]:
                self.scorecardUseLaunchTerminal.updateScore(1, True, associatedUUIDs=[launchTerminal.uuid], associatedNotes=f"The launch terminal (UUID: {launchTerminal.uuid}) has been used.")

        if not self.scorecardEnterOrbitVelocity.completed and "launchConfirmed" in launchTerminal.attributes["states"]:
            wiggleRoom = 2    # Allow 2 m/s of wiggle room in getting the value correct
            associatedNotes = []
            associatedUUIDs = []
            if abs(launchTerminal.attributes['orbitSpeed'] - launchTerminal.targetOrbitSpeed) < wiggleRoom:
                associatedUUIDs.append((launchTerminal.uuid,))
                associatedNotes.append(f"{launchTerminal.name} (UUID: {launchTerminal.uuid}) has correct orbital velocity {launchTerminal.attributes['orbitSpeed']}.")
                self.scorecardEnterOrbitVelocity.updateScore(1, True, associatedUUIDs, associatedNotes)
            else:
                associatedUUIDs.append((launchTerminal.uuid,))
                associatedNotes.append(f"{launchTerminal.name} (UUID: {launchTerminal.uuid}) has incorrect orbital velocity {launchTerminal.attributes['orbitSpeed']}.")
                self.scorecardEnterOrbitVelocity.updateScore(0, True, associatedUUIDs, associatedNotes)

        if self.scoringInfo["difficulty"] == "challenge":
            if not self.scorecardEnterFuelAndAmount.completed and "launchConfirmed" in launchTerminal.attributes["states"]:
                wiggleRoom = 2    # Allow 2L of wiggle room in getting the value correct
                associatedNotes = []
                associatedUUIDs = []
                selectedRightFuel = launchTerminal.attributes['fuel'] == launchTerminal.targetFuelType
                selectedRightAmount = abs(launchTerminal.attributes['fuelAmount'] - launchTerminal.targetFuelAmount) < wiggleRoom
                if selectedRightFuel and selectedRightAmount:
                    associatedUUIDs.append((launchTerminal.uuid,))
                    associatedNotes.append(f"{launchTerminal.name} (UUID: {launchTerminal.uuid}) has correct fuel ({launchTerminal.attributes['fuel']}) and amount ({launchTerminal.attributes['fuelAmount']}).")
                    self.scorecardEnterFuelAndAmount.updateScore(1, True, associatedUUIDs, associatedNotes)
                elif not selectedRightFuel:
                    associatedUUIDs.append((launchTerminal.uuid,))
                    associatedNotes.append(f"{launchTerminal.name} (UUID: {launchTerminal.uuid}) has incorrect fuel ({launchTerminal.attributes['fuel']}).")
                    self.scorecardEnterFuelAndAmount.updateScore(0, True, associatedUUIDs, associatedNotes)
                elif not selectedRightAmount:
                    associatedUUIDs.append((launchTerminal.uuid,))
                    associatedNotes.append(f"{launchTerminal.name} (UUID: {launchTerminal.uuid}) has incorrect fuel amount ({launchTerminal.attributes['fuelAmount']}).")
                    self.scorecardEnterFuelAndAmount.updateScore(0, True, associatedUUIDs, associatedNotes)

        if self.scoringInfo["difficulty"] != "easy":
            # Check if player has measured light angle at two different locations.
            if not self.scorecardMeasureLightAngle.completed:
                speedSquare = self.scoringInfo["speedSquare"][0]  # TODO: deal with multiple speed squares
                if len(speedSquare.collectedMeasurements) == 1:
                    self.scorecardMeasureLightAngle.updateScore(1, False, associatedUUIDs=[speedSquare.uuid], associatedNotes=f"The speed square (UUID: {speedSquare.uuid}) was used to measure light angle on one location.")
                elif len(speedSquare.collectedMeasurements) >= 2:
                    self.scorecardMeasureLightAngle.updateScore(2, True, associatedUUIDs=[speedSquare.uuid], associatedNotes=f"The speed square (UUID: {speedSquare.uuid}) was used to measure light angle on a at least two locations.")

            # Check if pendulum is activated.
            if not self.scorecardActivatePendulum.completed:
                pendulum = self.scoringInfo["pendulum"]
                if pendulum.attributes["isActivated"]:
                    self.scorecardActivatePendulum.updateScore(1, True, associatedUUIDs=[pendulum.uuid], associatedNotes=f"Pendulum (UUID: {pendulum.uuid}) has been activated.")

            # Check if pendulum has been read.
            if not self.scorecardReadoutPendulum.completed:
                pendulum = self.scoringInfo["pendulum"]
                for agent in self.world.agents:
                    if agent.actionHistory.queryActionObjects(ActionType.READ, arg1=pendulum, stopAtFirst=True):
                        self.scorecardReadoutPendulum.updateScore(1, True, associatedUUIDs=[agent.uuid, pendulum.uuid], associatedNotes=f"Agent (UUID: {agent.uuid}) has read the pendulum (UUID: {pendulum.uuid}).")

        if self.scoringInfo["difficulty"] == "challenge":
            loadCell = self.scoringInfo["loadCell"]

            # Check if player has used the load cell.
            if not self.scorecardUseLoadCellInterface.completed:
                if "interactedWith" in loadCell.attributes["states"]:
                    self.scorecardUseLoadCellInterface.updateScore(1, True, associatedUUIDs=[loadCell.uuid], associatedNotes=f"The load cell interface (UUID: {loadCell.uuid}) has been used.")

            # Try the different fuel
            if not self.scorecardTryDifferentFuel.completed:
                self.scorecardTryDifferentFuel.updateScore(len(loadCell.thrustMeasured), len(loadCell.thrustMeasured) == 3, associatedUUIDs=[loadCell.uuid], associatedNotes=f"Different fuel types have been tried with the load cell (UUID: {loadCell.uuid}).")

        # Update score
        self.score = sum(element.score for element in self.scoreCard)

        # Check whether the task is complete
        # Here, the task is complete if the right target orbital velocity has been entered in the launch terminal.
        self.completed = False
        self.completedSuccessfully = False
        if self.scoringInfo["difficulty"] != "challenge":
            if self.scorecardEnterOrbitVelocity.completed:
                self.completed = True
                self.completedSuccessfully = (self.scorecardEnterOrbitVelocity.score == 1)
        else:
            if self.scorecardEnterOrbitVelocity.completed and self.scorecardEnterFuelAndAmount.completed:
                self.completed = True
                self.completedSuccessfully = (self.scorecardEnterOrbitVelocity.score == 1 and self.scorecardEnterFuelAndAmount.score == 1)


class LaunchTerminal(NPCDevice):
    def __init__(self, world):
        Agent.__init__(self, world, "computer", "computer", defaultSpriteName="instruments_spectrometer")

        self.spriteCharacterPrefix = ""         # Disable the character prefix for this object (just use the default sprite)

        self.targetOrbitSpeed = None
        self.targetFuelAmount = None
        self.targetFuelType = None
        self.attributes["orbitSpeed"] = 0
        self.attributes["fuel"] = None
        self.attributes["fuelAmount"] = 0
        self.attributes["manualMaterialNames"] = ["Metal"]

    def checkOrbitSpeed(self):
        # Make sure the orbital speed is within bounds
        if (self.attributes['orbitSpeed'] < 0):
            self.attributes['orbitSpeed'] = 0
        elif (self.attributes['orbitSpeed'] > 25000):
            self.attributes['orbitSpeed'] = 25000

    def selectFuelA(self):
        self.attributes["fuel"] = "A"

    def selectFuelB(self):
        self.attributes["fuel"] = "B"

    def selectFuelC(self):
        self.attributes["fuel"] = "C"

    def checkFuelAmount(self):
        # Make sure the fuel amount is within bounds
        if (self.attributes['fuelAmount'] < 0):
            self.attributes['fuelAmount'] = 0
        elif (self.attributes['fuelAmount'] > 10000):
            self.attributes['fuelAmount'] = 10000

    def confirmLaunch(self):
        self.addState("launchConfirmed")

    def tick(self):
        self.checkOrbitSpeed()
        super().tick()

    def inferSpriteName(self, force:bool=False):
        # This will be the next last sprite name (when we flip the backbuffer)
        self.curSpriteName = "instruments_spectrometer"
        self.tempLastSpriteName = self.curSpriteName


def mkDialogLaunchTerminal(launchTerminal, difficulty):
    tree = DialogTree(launchTerminal)

    infos = "\n\nOrbital velocity currently set to {orbitSpeed} m/s.\nFuel type is {fuel}.\nFuel amount is {fuelAmount}L."

    rootNode = DialogNode("rootNode", f"-= Launch Terminal - Main menu =-{infos}", statesToAdd=["interactedWith"])
    rootNode.addDialogOption("Change target orbital velocity.", "setOrbitVelocityNode")
    if difficulty == "challenge":
        rootNode.addDialogOption("Change fuel type.", "setFuelTypeNode")
        rootNode.addDialogOption("Change fuel amount.", "setFuelAmountNode")

    rootNode.addDialogOption("Start countdown!", "startCountdownNode")
    rootNode.addDialogOption("Exit", "endNode")
    tree.addNode(rootNode)
    tree.setRoot(rootNode.name)

    startCountdownNode = DialogNode("startCountdownNode", f"-= Launch Terminal - Start Rocket Launch Countdown? =-{infos}\n\n(Once you confirmed, there's no coming back.)")
    startCountdownNode.addDialogOption("Confirm", "endNode", callback=launchTerminal.confirmLaunch)
    startCountdownNode.addDialogOption("Cancel", "rootNode")
    tree.addNode(startCountdownNode)

    setOrbitVelocityNode = DialogNode("setOrbitVelocityNode", "-= Launch Terminal - Setting Orbital Velocity =-\n\nOrbital velocity currently set to {orbitSpeed} m/s.\n(Must be between 0 and 25,000 m/s).", statesToAdd = [], statesToRemove = [])
    # Increase target velocity
    setOrbitVelocityNode.addDialogOption("Increase target velocity by 1000 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": 1000}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Increase target velocity by 100 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": 100}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Increase target velocity by 10 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": 10}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Increase target velocity by 1 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": 1}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    # Decrease target velocity
    setOrbitVelocityNode.addDialogOption("Decrease target velocity by 1000 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": -1000}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Decrease target velocity by 100 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": -100}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Decrease target velocity by 10 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": -10}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)
    setOrbitVelocityNode.addDialogOption("Decrease target velocity by 1 m/s", "setOrbitVelocityNode", floatVariablesToModify={"orbitSpeed": -1}, minMaxRange={"orbitSpeed": {"min": 0, "max": 25000}}, callback=launchTerminal.checkOrbitSpeed)

    setOrbitVelocityNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(setOrbitVelocityNode)

    setFuelTypeNode = DialogNode("setFuelTypeNode", "-= Launch Terminal - Setting Orbital Velocity =-\n\nCurrent fuel type is {fuel}.")
    setFuelTypeNode.addDialogOption(f"Fuel A", "setFuelTypeNode", callback=launchTerminal.selectFuelA)
    setFuelTypeNode.addDialogOption(f"Fuel B", "setFuelTypeNode", callback=launchTerminal.selectFuelB)
    setFuelTypeNode.addDialogOption(f"Fuel C", "setFuelTypeNode", callback=launchTerminal.selectFuelC)
    setFuelTypeNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(setFuelTypeNode)

    setFuelAmountNode = DialogNode("setFuelAmountNode", "-= Launch Terminal - Setting Fuel Amount =-\n\nFuel amount currently set to {fuelAmount}L.\n(Must be between 0 and 10,000L).")
    # Increase fuel
    setFuelAmountNode.addDialogOption("Increase fuel by 1000L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": 1000}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Increase fuel by 100L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": 100}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Increase fuel by 10L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": 10}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Increase fuel by 1L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": 1}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    # Decrease fuel
    setFuelAmountNode.addDialogOption("Decrease fuel by 1000L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": -1000}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Decrease fuel by 100L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": -100}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Decrease fuel by 10L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": -10}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)
    setFuelAmountNode.addDialogOption("Decrease fuel by 1L", "setFuelAmountNode", floatVariablesToModify={"fuelAmount": -1}, minMaxRange={"fuelAmount": {"min": 0, "max": 10000}}, callback=launchTerminal.checkFuelAmount)

    setFuelAmountNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(setFuelAmountNode)

    # OK node
    endNodeOK = DialogNode("endNode", "Exiting launch terminal.")
    tree.addNode(endNodeOK)

    # Store dialog tree in agent
    launchTerminal.setDialogTree(tree)


class LoadCellInterface(NPCDevice):
    def __init__(self, world):
        Agent.__init__(self, world, "load cell interface", "load cell interface", defaultSpriteName=f"instruments2_load_cell_interface")

        self.spriteCharacterPrefix = ""         # Disable the character prefix for this object (just use the default sprite)

        self.attributes["isPassable"] = False
        self.attributes["manualMaterialNames"] = ["Glass"]

        self.rocket = ()
        self.fuels = {}
        self.rocketDryMass = 0
        self.rocketTanksVolume = 0
        self.weight = 0
        self.thrustMeasured = set()

        self.addState("fuelEmpty")
        self.attributes["fuel"] = "None"
        self.attributes["thrust"] = 0
        self.attributes["duration"] = 0
        self.attributes["weight"] = 0

        self.nbTicksSinceActivation = 0

    def inferSpriteName(self, force:bool=False):
        # This will be the next last sprite name (when we flip the backbuffer)
        self.curSpriteName = "instruments2_load_cell_interface"
        self.tempLastSpriteName = self.curSpriteName

    def flush(self):
        self.attributes["fuel"] = "None"
        self.update()

    def selectA(self):
        self.attributes["fuel"] = "A"
        self.update()

    def selectB(self):
        self.attributes["fuel"] = "B"
        self.update()

    def selectC(self):
        self.attributes["fuel"] = "C"
        self.update()

    def update(self):
        fuel = self.attributes["fuel"]
        self.weight = self.rocketDryMass
        if fuel != "None":
            fuelMass = self.fuels[fuel]["density"] * self.rocketTanksVolume
            self.weight += fuelMass
            self.removeState("fuelEmpty")

        self.attributes["weight"] = np.round(self.weight, 1)
        self.attributes["duration"] = 0
        self.attributes["thrust"] = 0

    def fire(self):
        self.attributes["duration"] = 0
        self.attributes["thrust"] = 0
        self.rocket[0].setFiring(True)
        self.rocket[1].setFiring(True)

    def abort(self):
        self.removeState("testInProgress")
        self.addState("testCompleted")
        self.rocket[0].setFiring(False)
        self.rocket[1].setFiring(False)

    def tick(self):
        super().tick()
        if "testInProgress" in self.attributes['states']:
            self.attributes["duration"] += 1

            # Burn fuel.
            fuel = self.attributes["fuel"]
            self.attributes["thrust"] = np.round(self.fuels[fuel]["thrust"] / 1000, 1)
            if self.attributes["thrust"] > 0:
                self.thrustMeasured.add(self.attributes["thrust"])
            self.weight -= self.fuels[fuel]["massFlowRate"]
            if self.weight <= self.rocketDryMass:
                self.weight = self.rocketDryMass
                self.removeState("testInProgress")
                self.addState("testCompleted")
                self.addState("fuelEmpty")
                self.rocket[0].setFiring(False)
                self.rocket[1].setFiring(False)

            self.attributes["weight"] = np.round(self.weight, 1)


def mkDialogLoadCellInterface(loadCell):
    tree = DialogTree(loadCell)

    infos = "\n\nSelected fuel: {fuel}\nCurrent weight: {weight} kg\nGenerated thrust: {thrust} kN\nDuration: {duration} ticks"

    rootNode = DialogNode("rootNode", f"-= Load Cell Interface - Main menu =-{infos}", statesToAdd=["interactedWith"])
    rootNode.addDialogOption("Refuel", "setFuelNode", antiStates=["testInProgress", "testCompleted", "fuelEmpty"])
    rootNode.addDialogOption("Refuel (currently empty)", "setFuelNode", antiStates=["testInProgress", "testCompleted"], requiresStates=["fuelEmpty"])
    rootNode.addDialogOption("Fire!", "startFiringNode", antiStates=["testInProgress", "testCompleted", "fuelEmpty"], callback=loadCell.fire)
    rootNode.addDialogOption("Refresh", "rootNode", requiresStates=["testInProgress"])
    rootNode.addDialogOption("Refresh", "testCompletedNode", requiresStates=["testCompleted"])
    rootNode.addDialogOption("Abort!", "testAbortedNode", requiresStates=["testInProgress"], callback=loadCell.abort)
    rootNode.addDialogOption("Exit", "endNode")
    tree.addNode(rootNode)
    tree.setRoot(rootNode.name)

    testCompletedNode = DialogNode("testCompletedNode", f"-= Load Cell Interface - Test completed =-{infos}", statesToRemove=["testCompleted"])
    testCompletedNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(testCompletedNode)

    testAbortedNode = DialogNode("testAbortedNode", f"-= Load Cell Interface - Test aborted =-{infos}", statesToRemove=["testCompleted"])
    testAbortedNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(testAbortedNode)


    startCountdownNode = DialogNode("startFiringNode", f"-= Load Cell Interface - Test rocket fired =-{infos}", statesToAdd = ["testInProgress"])
    startCountdownNode.addDialogOption("Test in progress...", "rootNode")
    tree.addNode(startCountdownNode)

    setFuelNode = DialogNode("setFuelNode", f"-= Load Cell Interface - Refueling =-{infos}")
    setFuelNode.addDialogOption("Flush fuel from test rocket's tanks", "setFuelNode", callback=loadCell.flush)
    setFuelNode.addDialogOption(f"Fill rocket with {loadCell.rocketTanksVolume}L of fuel A", "rootNode", callback=loadCell.selectA)
    setFuelNode.addDialogOption(f"Fill rocket with {loadCell.rocketTanksVolume}L of fuel B", "rootNode", callback=loadCell.selectB)
    setFuelNode.addDialogOption(f"Fill rocket with {loadCell.rocketTanksVolume}L of fuel C", "rootNode", callback=loadCell.selectC)
    setFuelNode.addDialogOption("[Back]", "rootNode")
    tree.addNode(setFuelNode)

    # OK node
    endNodeOK = DialogNode("endNode", "Exiting Load Cell Interface.")
    tree.addNode(endNodeOK)

    # Store dialog tree in agent
    loadCell.setDialogTree(tree)
