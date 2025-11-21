# ObjectLoader.py
# This class loads objects from a spreadsheet in TSV format.
# The spreadsheet columns are largely object properties that the objects should have (in their attributes dictionaries).
# The first several columns are set meta-information: ReferenceName, ObjectName, objectType, contents.
# The first row is a header row, and is ignored.
# The purpose of the spreadsheet is to make it very easy to load new objects into the game.

import os
import inspect
from os.path import join as pjoin

from discoveryworld.objects import *
from discoveryworld.Agent import *

from discoveryworld.constants import DATA_PATH
from discoveryworld.scenarios.plant_growing import SoilControllerEasy
from discoveryworld.scenarios.space_sick_easy import SimplifiedScientificInstrument


class ObjectMaker:
    # Constructor
    def __init__(self, _dataPath, _filenameObjs, _filenameMaterials, world, knownSpriteNames=None):
        self.filenameObjs = pjoin(_dataPath or DATA_PATH, _filenameObjs)
        self.filenameMaterials = pjoin(_dataPath or DATA_PATH, _filenameMaterials)
        self.objects = {}
        self.warnings = []
        self.errors = []

        self.referentToLineNum = {}     # To help make error messages more specific

        # Store the reference to the World, so we don't need to ask for this every time we create an object
        self.world = world

        # Load material properties from index
        self.materialProperties = self.loadMaterialPropertyIndex(self.filenameMaterials)

        # Load object properties from index
        self.objectProperties = self.loadObjectPropertyIndex(self.filenameObjs)

        # Index of object classes
        # self.classIndex = self.createClassIndex()


        # Assign objects the materials they specified
        for key in self.objectProperties:
            materialNames = sorted(list(set(self.objectProperties[key]["materials"])))

            materials = []
            for materialName in materialNames:
                if (materialName in self.materialProperties):
                    materials.append(self.materialProperties[materialName])
                else:
                    self.errors.append("Object: " + key + " on line " + str(self.referentToLineNum[key]) + " has a material name: '" + materialName + "' that is not in the list of known material names.")
            self.objectProperties[key]["materials"] = materials


        # Secondary tests

        # # Check for unknown base classes.
        # unknownClassError = False
        # for key in self.objectProperties:
        #     baseClass = self.objectProperties[key]["BaseClass"]
        #     # Check to see if base class is in either the objectProperties or classIndex dictionaries
        #     if (baseClass not in self.objectProperties) and (baseClass not in self.classIndex):
        #         self.errors.append("Unknown base class: " + baseClass + " in object: " + key + " on line " + str(self.referentToLineNum[key]) + ".  Must be either an existing base class in the code, or a ReferenceName of another object class in the properties file.")
        #         unknownClassError = True

        # if (unknownClassError):
        #     # Provide a helpful message.
        #     self.errors.append("REMINDER: Known base class names: " + str(sorted(list(self.classIndex.keys()))))
        #     self.errors.append("REMINDER: Known object names: " + str(sorted(list(self.objectProperties.keys()))))


        # Check for objects that define "Object" as their base class, but have a blank 'defaultSpriteName'
        for key in self.objectProperties:
            baseClass = self.objectProperties[key]["BaseClass"]
            if (baseClass == "Object") and (self.objectProperties[key]["defaultSpriteName"] == ""):
                self.errors.append("Object: " + key + " on line " + str(self.referentToLineNum[key]) + " has base class 'Object', but has a blank 'defaultSpriteName'.  New objects (that override the base class Object) must specify a sprite name.")

        # Check any sprite names exist
        if knownSpriteNames is not None:
            unknownSpriteError = False
            for key in self.objectProperties:
                spriteName = self.objectProperties[key]["defaultSpriteName"]
                if (spriteName != "") and (spriteName not in knownSpriteNames):
                    self.errors.append("Object: " + key + " on line " + str(self.referentToLineNum[key]) + " has a sprite name: '" + spriteName + "' that is not in the list of known sprite names.")
                    unknownSpriteError = True
            if (unknownSpriteError):
                # Provide a helpful message.
                self.errors.append("REMINDER: Known sprite names: " + str(sorted(list(knownSpriteNames))))


        # Error reporting

        # Print warnings
        if len(self.warnings) > 0:
            print("\nWarnings (" + str(len(self.warnings)) + "):")
            for idx, warning in enumerate(self.warnings):
                print(str(idx) + ": " + warning)

        # Print errors
        if len(self.errors) > 0:
            print("\nErrors (" + str(len(self.errors)) + "):")
            for idx, error in enumerate(self.errors):
                print(str(idx) + ": " + error)

            print("\nErrors are present when loading objects. Exiting...")
            exit(1)


    #
    #   Creating new instances of objects
    #
    def createObject(self, objectReferenceName, *args, **kwargs):
        # Step 1: Check to see if the objectReferenceName is in the objectProperties dictionary, or whether it's a bare class name in the classIndex.
        if (objectReferenceName in self.objectProperties):
            # This object is in the objectProperties dictionary. Create a new instance of the object, then set its properties.
            # print("Object reference name: " + objectReferenceName + " is in the objectProperties dictionary.")
            # Create a new instance of the base class
            baseClass = self.objectProperties[objectReferenceName]["BaseClass"]
            # Create a new instance of the class
            obj = self._createObjectInstance(baseClass, *args, **kwargs)

            # Populate with general properties
            specialProperties = ["referencename", "objectname", "objecttype", "baseclass", "defaultSpriteName", "parts", "contents"]
            for key in self.objectProperties[objectReferenceName]:
                if (key.lower not in specialProperties):
                    # Set the property
                    value = self.objectProperties[objectReferenceName][key]
                    obj.attributes[key] = value

            # Populate with special properties
            # TODO: ReferenceName
            obj.name = self.objectProperties[objectReferenceName]["ObjectName"]
            obj.type = self.objectProperties[objectReferenceName]["ObjectType"]
            overrideSpriteName = self.objectProperties[objectReferenceName]["defaultSpriteName"]
            if (overrideSpriteName != ""):
                obj.defaultSpriteName = overrideSpriteName

            # Contents -- create any content objects that are required
            contents = self.objectProperties[objectReferenceName]["contents"]
            for cObjName in contents:
                # Create an instance of the contents object
                if (cObjName != ""):
                    cObjInstance = self.createObject(cObjName)
                    # Add it to this object
                    obj.addObject(cObjInstance, force=True)

            # Parts -- same as above
            parts = self.objectProperties[objectReferenceName]["parts"]
            for pObjName in parts:
                # Create an instance of the contents object
                if (pObjName != ""):
                    pObjInstance = self.createObject(pObjName)
                    # Add it to this object
                    obj.addPart(pObjInstance)

            # Special material initialization (if applicable)
            obj.initializeMaterialProperties(materialIndexDict=self.materialProperties)

            # Return
            return obj

        # Otherwise, assume it's a class name
        #elif (objectReferenceName in self.classIndex):
        else:
            # This object is in the classIndex dictionary. Create a new instance of the object, but do not add any new properties.
            # print("Object reference name: " + objectReferenceName + " is in the classIndex dictionary.")
            className = objectReferenceName
            # Create a new instance of the class
            obj = self._createObjectInstance(className, *args, **kwargs)

            if (obj is None):
                print("Error: Could not create object instance for object name: " + className)
                exit(1)

            # Special material initialization (if applicable)
            obj.initializeMaterialProperties(materialIndexDict=self.materialProperties)

            return obj

        # else:
        #     # Unknown object
        #     # Throw error
        #     print("Unknown reference name: '" + objectReferenceName + "'")
        #     exit(1)
        #     return None



        return None

    #   Create object instance
    #   This is a helper function that creates a new instance of an object, and sets its properties.
    #   Safer than using eval() to create a new instance of an object.
    def _createObjectInstance(self, className, *args, **kwargs):
        objectClasses = {
            "Object": Object,
            "Wall": Wall,
            "Floor": Floor,
            "CaveWall": CaveWall,
            "CaveFloor": CaveFloor,
            "Door": Door,
            "Sign": Sign,
            "SignVillage": SignVillage,
            "Grass": Grass,
            "Path": Path,
            "SoilTile": SoilTile,
            "Table": Table,
            "TableBedside": TableBedside,
            "Chair": Chair,
            "Bed": Bed,
            "Stove": Stove,
            "Sink": Sink,
            "Fridge": Fridge,
            "Microscope": Microscope,
            "PlantGeneric": PlantGeneric,
            "PlantRandomSmall": PlantRandomSmall,
            "Mushroom": Mushroom,
            "Mold": Mold,
            "Statue": Statue,
            "Fence": Fence,
            "Pot": Pot,
            "Jar": Jar,
            "Shovel": Shovel,
            "Seed": Seed,
            "Dirt": Dirt,
            "FlowerPot": FlowerPot,
            "Spectrometer": Spectrometer,
            "Thermometer": Thermometer,
            "PHMeter": PHMeter,
            "Sampler": Sampler,
            "RadiationMeter": RadiationMeter,
            "NPKMeter": NPKMeter,
            "PetriDish": PetriDish,
            "FertilizerPellet": FertilizerPellet,
            "FertilizerBag": FertilizerBag,
            "ChemicalDispenser": ChemicalDispenser,
            "BottleCleaner": BottleCleaningDevice,
            "Key": Key,
            "KeyRustyParametric": KeyRustyParametric,
            "AncientArtifact": AncientArtifact,
            "Flag": Flag,
            "RadioCarbonMeter": RadioCarbonMeter,
            "RadioisotopeMeter": RadioisotopeMeter,
            "ArtifactStoneHammer": ArtifactStoneHammer,
            "ArtifactBronzeChisel": ArtifactBronzeChisel,
            "ArtifactIronTongs": ArtifactIronTongs,
            "SoilController": SoilController,
            "PlantTreeBig": PlantTreeBig,
            "SoilNutrientMeter": SoilNutrientMeter,
            "SeedRequiringNutrients": SeedRequiringNutrients,
            "QuantumCrystal": QuantumCrystal,
            "CrystalReactor": CrystalReactor,
            "GeneratorSideLeft": GeneratorSideLeft,
            "GeneratorSideRight": GeneratorSideRight,
            "GeneratorCenter": GeneratorCenter,
            "Densitometer": Densitometer,
            "Desk": Desk,
            "FlagPole": FlagPole,
            "TableWithSign": TableWithSign,
            "CountingComputer": CountingComputer,
            "Coin": Coin,
            "Bookcase": Bookcase,
            "FloppyDisk": FloppyDisk,
            "Stick": Stick,
            "MeasuringTape": MeasuringTape,
            "ProteomicsMeter": ProteomicsMeter,
            "SoilControllerEasy": SoilControllerEasy,
            "MushroomDirectlyPoisonousRandom": MushroomDirectlyPoisonousRandom,
            "SeedDirectlyPoisonousMushroom": SeedDirectlyPoisonousMushroom,
            "GlowingRockDetector": GlowingRockDetector,

            # Colored objects
            "PaintBucket": PaintBucket,
            "ColoredMushroom": ColoredMushroom,
            "ColoredFlower": ColoredFlower,
            "ColoredKey": ColoredKey,

            # Seeds # possibleNutrients = ["potassium", "titanium", "lithium", "thorium", "barium"]
            "SeedNutrientPot1": [SeedRequiringNutrients, {"potassium": 1}],
            "SeedNutrientPot2": [SeedRequiringNutrients, {"potassium": 2}],
            "SeedNutrientPot3": [SeedRequiringNutrients, {"potassium": 3}],
            "SeedNutrientTit1": [SeedRequiringNutrients, {"titanium": 1}],
            "SeedNutrientTit2": [SeedRequiringNutrients, {"titanium": 2}],
            "SeedNutrientTit3": [SeedRequiringNutrients, {"titanium": 3}],
            "SeedNutrientLit1": [SeedRequiringNutrients, {"lithium": 1}],
            "SeedNutrientLit2": [SeedRequiringNutrients, {"lithium": 2}],
            "SeedNutrientLit3": [SeedRequiringNutrients, {"lithium": 3}],
            "SeedNutrientTho1": [SeedRequiringNutrients, {"thorium": 1}],
            "SeedNutrientTho2": [SeedRequiringNutrients, {"thorium": 2}],
            "SeedNutrientTho3": [SeedRequiringNutrients, {"thorium": 3}],
            "SeedNutrientBar1": [SeedRequiringNutrients, {"barium": 1}],
            "SeedNutrientBar2": [SeedRequiringNutrients, {"barium": 2}],
            "SeedNutrientBar3": [SeedRequiringNutrients, {"barium": 3}],
            "SeedRequiringNutrientsRuleBased": SeedRequiringNutrientsRuleBased,

            # Substances
            "TestSubstance": [Substance, "testSubstance"],
            "PurpleSubstance": [Substance, "purpleSubstance"],
            "substanceCleaner": [Substance, "cleaner"],
            "SubstanceA": [Substance, "Substance A"],
            "SubstanceB": [Substance, "Substance B"],
            "SubstanceC": [Substance, "Substance C"],
            "SubstanceD": [Substance, "Substance D"],

            "Sand": Sand,
            "LaunchPad": LaunchPad,
            "Rocket": Rocket,
            "Cactus": Cactus,
            "BigCactus": BigCactus,
            "SandPath": SandPath,
            "LaunchMonitor": LaunchMonitor,
            "FuelTank": FuelTank,
            "SpeedSquare": SpeedSquare,
            "Pendulum": Pendulum,
            "RocketryBook": RocketryBook,
            "LoadCell": LoadCell,
            "LoadCellWall": LoadCellWall,

            # Simplified Scientific Instruments
            "SimplifiedMicroscope": [SimplifiedScientificInstrument, "microscope"],
            "SimplifiedSpectrometer": [SimplifiedScientificInstrument, "spectrometer"],
            "SimplifiedPHMeter": [SimplifiedScientificInstrument, "ph_meter"],
            "SimplifiedRadiationMeter": [SimplifiedScientificInstrument, "radiation_meter"],
            "SimplifiedDensitometer": [SimplifiedScientificInstrument, "densitometer"],

        }


        # Check that the class name is in the dictionary
        if (className not in objectClasses):
            # Unknown class -- throw error
            print("Unknown class: " + className)
            return None
            #exit(1)

        # Special case: Object (generic)
        if (className == "Object"):
            obj = Object(self.world, "", "", "")     # The name, type, and default sprite should get updated by the calling function
            return obj


        # Check if the class has arguments
        if (type(objectClasses[className]) is list):
            # Special case: Substance
            # Create a new instance of the class
            # 1 argument
            if (len(objectClasses[className]) == 2):
                obj = objectClasses[className][0](self.world, objectClasses[className][1])
                return obj
            # 2 arguments
            elif (len(objectClasses[className]) == 3):
                obj = objectClasses[className][0](self.world, objectClasses[className][1], objectClasses[className][2])
                return obj
            # 3 arguments
            elif (len(objectClasses[className]) == 4):
                obj = objectClasses[className][0](self.world, objectClasses[className][1], objectClasses[className][2], objectClasses[className][3])
                return obj
            # 4+ unsupported
            else:
                print("Unsupported number of arguments for class: " + className)
                exit(1)
        else:
            # General case: Create a new instance of the class
            obj = objectClasses[className](self.world, *args, **kwargs)
            return obj




    # Create an index of all the Classes that are in the file 'ObjectModel.py'
    # Store them in a dictionary, where the key is the class name and the value is the class itself.
    # This is used to create new instances of objects
    # The imports needed for below are: import inspect
    def createClassIndex(self):
        # Get the path to the ObjectModel.py file
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/ObjectModel.py"

        # Load the file
        with open(path) as f:
            lines = f.readlines()

        # Find the line that starts with "class" and ends with ":"
        # The class name is the word after "class" and before ":"
        # Store the class name and the class itself in a dictionary
        # Note: This is a very simple way to do this. It will break if the class definition is spread over multiple lines, or if the class name is not the first word after "class"
        classIndex = {}
        for line in lines:
            if line.strip().startswith("class") and line.strip().endswith(":"):
                className = line.strip().split(" ")[1].split(":")[0]
                # Remove anything after "("
                classNameShort = className.split("(")[0]
                # Store
                classIndex[classNameShort] = className

        return classIndex



    #
    #   Loading properties spreadsheet
    #

    # Load object properties from spreadsheet
    # The spreadsheet is a TSV file that contains a list of objects and their properties
    def loadObjectPropertyIndex(self, filenameObjs):
        print("Loading object properties from index file: " + filenameObjs)

        # Load the TSV index file
        with open(filenameObjs) as f:
            lines = f.readlines()

        # Store the header
        header = lines[0].strip().split("\t")

        # For each object line, parse the line (as a dictionary, with the key the column header and the value the column value)
        out = {}
        lineNum = 1
        for line in lines[1:]:
            lineNum += 1

            # Skip the line if it's blank, of if it starts with a "#" (comment)
            if (line.strip() == "") or (line.strip()[0] == "#"):
                continue

            # Parse the line into a dictionary of key-value pairs (raw)
            keyValuePairs = dict(zip(header, line.split("\t")))

            # If the reference name is blank, skip this line and add a warning
            referenceName = keyValuePairs["ReferenceName"]
            if (keyValuePairs["ReferenceName"] == ""):
                # Reference name is blank -- skip
                self.warnings.append("Warning: Objects: ReferenceName is blank for line " + str(lineNum) + ". Skipping this line.")
                continue

            # Convert the values to the appropriate type
            keyValuePairsOut = {}
            for key in keyValuePairs.keys():
                newKeyName = key.split("_")[0]
                keyValuePairsOut[newKeyName] = self.parseToTypeHelper(key, keyValuePairs[key], lineNum=lineNum)

            # Store

            # Check for reference name already existing
            if (referenceName in out.keys()):
                self.errors.append("Error: Objects: ReferenceName '" + referenceName + "' already exists in the object index.  Reference names must be unique. (line " + str(lineNum) + ")")

            else:
                # Reference name is unique -- store
                out[referenceName] = keyValuePairsOut

            self.referentToLineNum[referenceName] = lineNum     # Store to make subsequent error reporting more helpful later on.


        print("Loaded properties for " + str(len(out)) + " total objects from " + filenameObjs + ".")

        # Return
        return out


    # Load material properties from spreadsheet
    # The spreadsheet is a TSV file that contains a list of materials and their properties
    def loadMaterialPropertyIndex(self, filenameObjs):
        print("Loading material properties from index file: " + filenameObjs)

        # Load the TSV index file
        with open(filenameObjs) as f:
            lines = f.readlines()

        # Store the header
        header = lines[0].strip().split("\t")

        # For each object line, parse the line (as a dictionary, with the key the column header and the value the column value)
        out = {}
        lineNum = 1
        for line in lines[1:]:
            lineNum += 1

            # Skip the line if it's blank, of if it starts with a "#" (comment)
            if (line.strip() == "") or (line.strip()[0] == "#"):
                continue

            # Parse the line into a dictionary of key-value pairs (raw)
            keyValuePairs = dict(zip(header, line.split("\t")))

            # If the reference name is blank, skip this line and add a warning
            referenceName = keyValuePairs["ReferenceName"]
            if (keyValuePairs["ReferenceName"] == ""):
                # Reference name is blank -- skip
                self.warnings.append("Warning: Materials: ReferenceName is blank for line " + str(lineNum) + ". Skipping this line.")
                continue

            # Convert the values to the appropriate type
            keyValuePairsOut = {}
            for key in keyValuePairs.keys():
                newKeyName = key.split("_")[0]
                keyValuePairsOut[newKeyName] = self.parseToTypeHelper(key, keyValuePairs[key], lineNum=lineNum)

            # Store

            # Check for reference name already existing
            if (referenceName in out.keys()):
                self.errors.append("Error: Materials: ReferenceName '" + referenceName + "' already exists in the material index.  Reference names must be unique. (line " + str(lineNum) + ")")

            else:
                # Reference name is unique -- store
                out[referenceName] = keyValuePairsOut

            self.referentToLineNum[referenceName] = lineNum     # Store to make subsequent error reporting more helpful later on.


        print("Loaded properties for " + str(len(out)) + " total materials from " + filenameObjs + ".")

        # Return
        return out

    # This helper function parses a string (a cell from the properties spreadsheet) into a specific type, based on a column header string ("name_type")
    def parseToTypeHelper(self, headerKeyStr, valueStr, lineNum = None):
        # Step 1: Determine the type of the value.
        # Headers are of the form "name_type", where "type" can be one of a number of formats.

        # Make sure there's at least one "_" in the header key (otherwise, default to 'str')
        typeName = "str"
        if ("_" in headerKeyStr):
            typeName = headerKeyStr.split("_")[-1].lower()

        # Step 2: Parse the value into the appropriate type
        # Known types: "list", "set", "str", "bool", "int", "float"
        knownTypes = ["list", "listfloat" "set", "str", "bool", "int", "float"]

        # Strip any whitespace from the value string
        valueStr = valueStr.strip()

        if (typeName == "list"):
            # Split the string by commas
            out = valueStr.split(",")

            # Strip whitespace from each element
            out = [x.strip() for x in out]

            # Return
            return out

        elif (typeName == "listfloat"):
            # First, check for an empty list
            if (valueStr.strip() == ""):
                return []

            # Split the string by commas
            out = valueStr.split(",")

            # Strip whitespace from each element
            out = [x.strip() for x in out]

            # Convert each element to a float
            try:
                out = [float(x) for x in out]
            except:
                self.errors.append("Error: Could not convert all the elements of this list to floats: " + str(out))
                return None

            # Return
            return out

        elif (typeName == "set"):
            # Split the string by commas
            out = valueStr.split(",")

            # Strip whitespace from each element
            out = [x.strip() for x in out]

            # Return
            return set(out)

        elif (typeName == "str"):
            # Return
            return valueStr

        elif (typeName == "bool"):
            # Return
            if (valueStr.lower() == "true"):
                return True
            elif (valueStr.lower() == "false"):
                return False
            else:
                self.errors.append("Error: Unknown bool value: '" + valueStr + "' for header key '" + headerKeyStr + "' on line " + str(lineNum) + ".")
                return None

        elif (typeName == "int"):
            # Try to convert to int
            try:
                out = int(valueStr)
                return out
            except:
                self.errors.append("Error: Unknown int value: '" + valueStr + "' for header key '" + headerKeyStr + "' on line " + str(lineNum) + ".")
                return None

        elif (typeName == "float"):
            # Try to convert to float
            try:
                out = float(valueStr)
                return out
            except:
                self.errors.append("Error: Unknown float value: '" + valueStr + "' for header key '" + headerKeyStr + "' on line " + str(lineNum) + ".")
                return None

        else:
            self.errors.append("Error: Unknown type '" + typeName + "' for header key '" + headerKeyStr + "'.  Recognized types: " + str(knownTypes) + ".")
            return None
