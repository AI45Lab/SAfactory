from functools import reduce
import math
from discoveryworld.ActionSuccess import ActionSuccess
from discoveryworld.objects.Object import Object


class BottleCleaningDevice(Object):
    """ (Container) Cleans all 'substances' out of a container """
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "bottle cleaner", "bottle cleaner", defaultSpriteName = "instruments_bottle_cleaner")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = True                       # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = False                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = False                 # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")
        self.attributes['contentsVisible2D'] = False               # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)

        # Can be used (as a dispenser, into a container)
        self.attributes["isUsable"] = True                        # Can it be used?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    #
    #   Actions (use with)
    def actionUseWith(self, patientObj):
        # Use this object on the patient object

        # Check if the patient object is a container
        if (not patientObj.attributes["isContainer"]):
            return ActionSuccess(False, "You can't use the bottle cleaner on something that isn't a container (" + str(patientObj.name) + ").")

        if (patientObj.attributes["isContainer"] == False):
            return ActionSuccess(False, "You can't use the bottle cleaner on something that isn't a container (" + str(patientObj.name) + ").")

        # Check that the container is open
        if (patientObj.attributes["isOpenContainer"] == False):
            return ActionSuccess(False, "You can't use the bottle cleaner on a container that isn't open (" + str(patientObj.name) + ").")

        # Search for any instances of 'substance' in the contents of the patient object
        substancesToRemove = []
        for obj in patientObj.contents:
            if ("isSubstance" in obj.attributes) and (obj.attributes["isSubstance"] == True):
                substancesToRemove.append(obj)

        # If there are no substances to remove, then return a failure
        if (len(substancesToRemove) == 0):
            return ActionSuccess(False, "The " + str(patientObj.name) + " has no substances to remove.")

        # If we reach here, the patient object is a container, it's open, and it contains substances.
        # Now we can remove the substances from the container (and indeed the world).
        for sub in substancesToRemove:
            patientObj.removeObject(sub)
            self.world.removeObject(sub)
            patientObj.invalidateSpritesThisWorldTile()
            self.invalidateSpritesThisWorldTile()

        # Return success
        return ActionSuccess(True, "You clean the " + str(patientObj.name) + " of all substances.")


    def tick(self):
        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # Always the same sprite name
        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class ChemicalDispenser(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "dispenser", "dispenser", defaultSpriteName = "instruments_chemical_dispenser")

        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = True                       # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = True                      # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = False                 # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")
        self.attributes['contentsVisible2D'] = False               # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)

        # Can be used (as a dispenser, into a container)
        self.attributes["isUsable"] = True                        # Can it be used?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Auto fill
        self.autoFillCheckForObjectName = None
        self.autoFillFillObjectName = None
        self.lastAddedCount = 0
        self.replenishTime = 5

    # If we want this object to autofill (i.e. perpetually fill with a supply of an object), set it here.
    def setAutoFill(self, checkObjectName:str, fillObjectName:str, minCount:int, replenishTime:int=8):
        self.autoFillCheckForObjectName = checkObjectName
        self.autoFillFillObjectName = fillObjectName
        self.autoFillMinCount = minCount
        self.replenishTime = replenishTime
        self.lastAddedCount = 0


    #
    #   Actions (use with)
    #   TODO
    def actionUseWith(self, patientObj):
        # Use this object on the patient object

        # Check if the patient object has the 'isShovelable' attribute
        if (not patientObj.attributes["isContainer"]):
            # Can't use the shovel on this object
            return ActionSuccess(False, "You can't use the chemical dispenser on something that isn't a container (" + str(patientObj.name) + ").")

        if (patientObj.attributes["isContainer"] == False):
            # Can't use the shovel on this object
            return ActionSuccess(False, "You can't use the chemical dispenser on something that isn't a container (" + str(patientObj.name) + ").")

        # Check that the container is open
        if (patientObj.attributes["isOpenContainer"] == False):
            return ActionSuccess(False, "You can't use the chemical dispenser on a container that isn't open (" + str(patientObj.name) + ").")

        # Check that this dispenser isn't empty
        if (len(self.contents) == 0):
            return ActionSuccess(False, "The chemical dispenser is empty.")


        # If we reach here, the patient object is a container, and it is open, and the dispenser is not empty.
        # We can now dispense a single object of the contents into the container.

        # Get the first object in the dispenser
        objToAdd = self.contents[0]
        # Add it to the patient object (container)
        self.world.removeObject(objToAdd)                   # Remove the object from the world
        patientObj.addObject(objToAdd, force=True)          # Add the object to the patient container
        # Invalidate the sprites for the container, and the dispenser
        patientObj.invalidateSpritesThisWorldTile()
        self.invalidateSpritesThisWorldTile()

        # If the autofill is set, then perform the autofill right now (after the object has been dispensed), so make sure there's always something in the dispenser.
        self.performAutoFill()

        # Return success
        return ActionSuccess(True, "You dispense " + str(objToAdd.name) + " into the " + str(patientObj.name) + ".")


    # Auto-fill the dispenser back up, if set
    def performAutoFill(self):
        #if (self.autoFillObjectName != None and self.autoFillMinCount != None):
        if (self.autoFillCheckForObjectName != None) and (self.autoFillFillObjectName != None) and (self.autoFillMinCount != None):
            # Count how much of the object is contained within the root level of the contents
            count = 0
            for obj in self.contents:
                if (obj.name == self.autoFillCheckForObjectName):
                    count += 1

            # Fill up to the minimum count
            if (count < self.autoFillMinCount):
                # Check to see if the last added count is zero
                if (self.lastAddedCount <= 0):
                    # Add one more
                    objToAdd = self.world.createObject(self.autoFillFillObjectName)
                    self.addObject(objToAdd, force=True)
                    # Tick the object to update its sprite
                    objToAdd.tick()
                    # Invalidate object sprite
                    self.needsSpriteNameUpdate = True
                    # Tick
                    self.lastAddedCount = self.replenishTime     # Wait a few ticks before adding the next one
                else:
                    self.lastAddedCount -= 1


    def tick(self):
        # Call superclass
        Object.tick(self)

        # Perform auto fill
        self.performAutoFill()





    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = "instruments_chemical_dispenser"

        # # Infer sprite based on whether empty/non-empty
        # if (len(self.contents) == 0):
        #     self.curSpriteName = "placeholder_jar_empty"
        # elif (len(self.contents) == 1):
        #     self.curSpriteName = "placeholder_jar_full1"
        # elif (len(self.contents) == 2):
        #     self.curSpriteName = "placeholder_jar_full2"
        # else:
        #     self.curSpriteName = "placeholder_jar_full3"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Substance(Object):
    # Constructor
    def __init__(self, world, substanceName:str="unknownSubstance"):
        # Default sprite name
        Object.__init__(self, world, "substance", "substance", defaultSpriteName = "instruments_sample")

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = False                     # Is it a container?
        self.attributes['isOpenable'] = False                      # Can not be opened (things are stored in the open pot)
        self.attributes['isOpenContainer'] = False                  # If it's a container, then is it open?
        self.attributes['containerPrefix'] = "in"                  # Container prefix (e.g. "in" or "on")
        self.attributes['contentsVisible2D'] = False               # If it is a container, do we render the contents in the 2D representation, or is that already handled (e.g. for pots/jars, that render generic contents if they contain any objects)

        # Substance properties
        self.attributes["substanceName"] = substanceName          # Name of the substance
        self.attributes['isSubstance'] = True                     # Is it a substance?
        self.attributes['isAutoReacting'] = True                  # Does it react automatically with other substances?
        self.attributes['mixtureDict'] = {}                       # Dictionary of substances and their proportions in the mixture

        # Material
        self.attributes["manualMaterialNames"] = ["SubstanceA"]  # Currently all the same, since these properties aren't really used in any existing scenario

    def normalizeSubstanceName(self, strIn:str):
        # Normalize the substance name -- remove any `substance name (X measures)` and just keep the substance name
        normalizedName = strIn
        if ("(" in strIn):
            normalizedName = strIn.split(" (")[0].strip()
        return normalizedName

    def tick(self):
        # Stop if tick has already been completed
        if (self.tickCompleted):
            return

        #print("## Substance tick: " + str(self.attributes["substanceName"]) + " (" + str(self.name) + ") UUID: " + str(self.uuid))

        # Call superclass
        Object.tick(self)

        # The substance name is dependent on the contents of the substance.

        oldSubstanceName = self.attributes["substanceName"]
        # Case 1: Empty substance
        if (len(self.contents) == 0):
            # Pure substance, keep current substance name
            #self.attributes["substanceName"] = "unknown substance"
            #self.attributes['mixtureDict'] = {self.attributes["substanceName"]: 1.0}
            self.attributes['mixtureDict'] = { self.normalizeSubstanceName(self.attributes["substanceName"]): 1.0}

            #print("SUBSTANCE TICK: Pure substance")
            pass

        # Case 2: Single substance
        elif (len(self.contents) == 1):
            # Check that the contents are a substance
            obj = self.contents[0]
            if (obj.attributes['isSubstance']):
                self.attributes["substanceName"] = obj.attributes["substanceName"]
            else:
                self.attributes["substanceName"] = "unknown substance"

            # Normalize the substance name -- remove any `substance name (X measures)` and just keep the substance name
            self.attributes['mixtureDict'] = { self.normalizeSubstanceName(self.attributes["substanceName"]): 1.0}

            #print("SUBSTANCE TICK: Only 1 substance")
        # Case 3: Multiple substances
        else:
            # First, create a frequency counter of the substances
            contentsFreq = {}
            for obj in self.contents:
                # Check if obj is a substance
                if (obj.attributes['isSubstance']):
                    subName = obj.attributes["substanceName"]
                    if (subName in contentsFreq):
                        contentsFreq[subName] += 1
                    else:
                        contentsFreq[subName] = 1

            #print("SUBSTANCE TICK: Multiple substances: " + str(contentsFreq) + " (old: " + str(oldSubstanceName) + ")")

            # Check if there are zero keys in the frequency counter
            if (len(contentsFreq) == 0):
                self.attributes["substanceName"] = "unknown substance"
                self.attributes['mixtureDict'] = {self.attributes["substanceName"]: 1.0}

            # Check if there is only one key in the frequency counter
            elif (len(contentsFreq) == 1):
                self.attributes["substanceName"] = list(contentsFreq.keys())[0] + " (" + str(contentsFreq[list(contentsFreq.keys())[0]]) + " measures)"
                self.attributes['mixtureDict'] = {self.normalizeSubstanceName(self.attributes["substanceName"]): 1.0}

            # If there are multiple keys in the frequency counter, then we need to create a mixture name
            else:
                # Next, sort by frequency
                sortedContentsFreq = sorted(contentsFreq.items(), key=lambda x: x[1], reverse=True)
                # Then, make a string of the form "X parts A, Y parts B, Z parts C, etc."
                self.attributes['mixtureDict'] = {}
                parts = []

                substanceName = "mixture ("

                for i in range(len(sortedContentsFreq)):
                    if (i > 0):
                        substanceName += ", "
                    name = str(sortedContentsFreq[i][0])
                    numParts = sortedContentsFreq[i][1]
                    substanceName += str(numParts) + " parts " + name
                    self.attributes['mixtureDict'][name] = numParts
                    parts.append(numParts)

                substanceName += ")"

                self.attributes["substanceName"] = substanceName

                # For the mixtureDict, try to determine if there's a lowest common denominator -- e.g. 5 parts X and 10 parts Y could be simplified to 1 part X and 2 parts Y
                # This is a bit tricky, because we need to find the greatest common divisor of all the parts.
                def findGCDList(listOfInts):
                    return reduce(math.gcd, listOfInts)
                gcd = findGCDList(parts)
                # If the gcd is greater than 1, then we can simplify the mixtureDict
                #print("GCD: " + str(gcd) + " (mixtureDict: " + str(self.attributes['mixtureDict']) + ")")
                if (gcd > 1):
                    for key in self.attributes['mixtureDict'].keys():
                        self.attributes['mixtureDict'][key] = self.attributes['mixtureDict'][key] / gcd
                    #print("Simplified mixtureDict: " + str(self.attributes['mixtureDict']))




        # Check if the sprite needs to be invalidated
        if (oldSubstanceName != self.attributes["substanceName"]):
            self.needsSpriteNameUpdate = True

        # Set the name of the object to the substance name
        self.name = self.attributes["substanceName"]

        # Self-reacting substances
        # Check if the substance is self-reacting
        if (self.attributes['isAutoReacting']):

            # Get a list of all substances in the same container as this substance (unless the container for this substance is also a Substance)
            substancesInContainer = []
            mixturesToPossiblyRemove = []       # Only remove if they're combined with another substance/mixture.
            numUnmixedFound = 0
            if (self.parentContainer is not None) and (self.parentContainer.attributes['isSubstance'] == False):
                for obj in self.parentContainer.contents:
                    if (obj.attributes['isSubstance']) and (obj.attributes["isAutoReacting"] == True):
                        # If it 'contains' nothing, then it's a pure substance
                        if (len(obj.contents) == 0):
                            substancesInContainer.append(obj)
                            numUnmixedFound += 1
                            #print("\t Looking for reactants: Pure substance found: " + str(obj.attributes["substanceName"]) + " (" + str(obj.name) + ", UUID: " + str(obj.uuid) + ")")
                        # If it contains something, then it's a mixture
                        else:
                            found = False
                            for sub in obj.contents:
                                if (sub.attributes['isSubstance']):
                                    substancesInContainer.append(sub)
                                    found = True
                            if (found):
                                numUnmixedFound += 1
                                mixturesToPossiblyRemove.append(obj)
                                #print("\t Looking for reactants: Mixture found: " + str(obj.attributes["substanceName"]) + " (" + str(obj.name) + ", UUID: " + str(obj.uuid) + ")")

            if (numUnmixedFound > 1):
                # Check to see that there is more than one different type of substance in the container (by 'substanceName' attribute)
                #substanceNames = set()
                #for sub in substancesInContainer:
                #    substanceNames.add(sub.attributes["substanceName"])

                # If the list of substances in the container is greater than 1, then we have a self-reacting substance
                #if (len(substanceNames) > 1):

                # DEBUG: Show a list of names of the substances that are about to react
                substancesNamesInContainer = [sub.attributes["substanceName"] for sub in substancesInContainer]
                #print("Self-reacting substance detected.  Reacting substances: " + str(substancesNamesInContainer))
                #print("UUIDs of reacting substances: " + str([sub.uuid for sub in substancesInContainer]))

                # The way reactions are handled is by making a new Substance ("reacting substance"), and adding all the substances in the container to it.
                reactingSubstance = Substance(self.world, "reacting substance")
                # Add the 'reacting substance' to the parent container
                self.parentContainer.addObject(reactingSubstance, force=True)
                # Add all the substances in the container to the reacting substance
                for sub in substancesInContainer:
                    # Set 'sub' to not tick (otherwise it will start spawning new substances in the world)
                    sub.tickCompleted = True
                    # Remove from current container
                    self.world.removeObject(sub)
                    # Add to reacting substance
                    reactingSubstance.addObject(sub, force=True)

                # Remove the mixtures that were combined with other substances
                for mix in mixturesToPossiblyRemove:
                    self.world.removeObject(mix)

                # (TODO: Also tick the reacting substances?)
                #print("Created new reacting substance (UUID: " + str(reactingSubstance.uuid) + ")")
                #print("Calling Tick() on new substance (reacting substance, UUID: " + str(reactingSubstance.uuid) + ")")
                #import time
                #time.sleep(1)
                # Call the tick on the reacting substance
                reactingSubstance.tick()



    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return
        # # Infer sprite based on whether empty/non-empty
        # if (len(self.contents) == 0):
        #     self.curSpriteName = "placeholder_jar_empty"
        # elif (len(self.contents) == 1):
        #     self.curSpriteName = "placeholder_jar_full1"
        # elif (len(self.contents) == 2):
        #     self.curSpriteName = "placeholder_jar_full2"
        # else:
        #     self.curSpriteName = "placeholder_jar_full3"

        self.curSpriteName = "instruments_sample"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class Key(Object):
    # Constructor
    def __init__(self, world, color=None, isRusted=True):
        # Default sprite name
        self.color = color
        name = "key" if self.color is None else f"{self.color} key"
        defaultSpriteName = "instruments_key" if self.color is None else f"instruments_key_{self.color}"
        Object.__init__(self, world, "key", name, defaultSpriteName)

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = False                    # Is it a container?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Rusted
        self.attributes['isRusted'] = isRusted                    # Is the key rusted?
        self.attributes['rustLevel'] = 3 if isRusted else 0       # Description of the rust (0=none, 1=light, 2=medium, 3=heavy)

        # Key ID
        self.attributes['keyID'] = 1                              # Key ID (1 by default)


    def setKeyID(self, keyID:int):
        self.attributes['keyID'] = keyID


    def tick(self):
        # Call superclass
        Object.tick(self)

        # Clean rust of key (if conditions are met)
        # If the key is in a container, and that container contains SUBSTANCE TODO, and it's rusted, then remove the rust.
        # Check if the key is in a container
        if (self.parentContainer is not None):
            # Get parent container contents
            parentContainerContents = self.parentContainer.contents
            # Check if the parent container contains a specific substance
            containsCleaner = False
            #print("### KEY: Checking for cleaner substance in parent container")
            partialEvidenceLevel = 0

            # Check to make sure there are not more than 1 subtances in the container
            numSubstances = 0
            for obj in parentContainerContents:
                if (obj.type == "substance"):
                    numSubstances += 1

            if (numSubstances > 1):
                #print("## More than 1 substance in container.  Not evaluating rust reduction further, since those subtances may react later.")
                return

            for obj in parentContainerContents:
                #print("## Key: Mixture dictionary: " + str(obj.attributes['mixtureDict']) + " (containsCleaner: " + str(containsCleaner) + ")")
                # Check for "Cleaner"
                if (obj.type == "substance") and (obj.attributes["substanceName"] == "cleaner"):
                    containsCleaner = True
                    break
                # Check for a specific mixture (1 part Substance A, 2 parts substance C)
                if (obj.type == "substance") and (len(obj.attributes['mixtureDict']) == 2):
                    #print("### In container with substance  (mixtureDict: " + str(obj.attributes['mixtureDict']) + ")")
                    if ("Substance A" in obj.attributes['mixtureDict']) and ("Substance C" in obj.attributes['mixtureDict']):
                        #print("### In container with Substance A and C")
                        if (obj.attributes['mixtureDict']["Substance A"] == 1) and (obj.attributes['mixtureDict']["Substance C"] == 2):
                            #print("### In container with Substance A and C, and the right proportions")
                            containsCleaner = True
                            break
                # Partial evidence -- give some help for hill-climbing
                if ("Substance A" in obj.attributes['mixtureDict']) and ("Substance B" not in obj.attributes['mixtureDict']):
                    partialEvidenceLevel += 1
                    #print("## Partial evidence (Substance A)")
                if ("Substance C" in obj.attributes['mixtureDict']) and ("Substance B" not in obj.attributes['mixtureDict']):
                    partialEvidenceLevel += 1
                    #print("## Partial evidence (Substance C)")

            # If the parent container contains the cleaner substance, then remove the rust from the key
            if (containsCleaner):
                self.attributes['isRusted'] = False
                self.attributes['rustLevel'] = 0    # No more rust
                # Invalidate the sprite
                self.needsSpriteNameUpdate = True

            # Handle partial evidence -- describe the key as less rusty, if certain combinations of substances are found
            #print("### Partial evidence level: " + str(partialEvidenceLevel))
            if (partialEvidenceLevel > 0):
                # If the partial evidence level is high enough, then remove the rust from the key
                newRustLevel = 3 - partialEvidenceLevel
                if (newRustLevel < self.attributes['rustLevel']):
                    self.attributes['rustLevel'] = newRustLevel
                    self.needsSpriteNameUpdate = True

        # Update key name based on whether it's rusted or not
        # If the key is rusted, then set it's name to key (rusted)
        if (self.attributes['isRusted']):# and (self.name != "rusted key"):
            if (self.attributes['rustLevel'] == 3):
                self.name = "rusted key (heavily rusted)"
            elif (self.attributes['rustLevel'] == 2):
                self.name = "rusted key (moderately rusted)"
            elif (self.attributes['rustLevel'] == 1):
                self.name = "rusted key (lightly rusted)"

        elif (not self.attributes['isRusted']):
            self.name = "key (no rust)"

        if self.color:
            self.name = f"{self.color} {self.name}"

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If rusted, use the rusty sprite
        if (self.attributes['isRusted']):
            self.curSpriteName = "instruments_key_rusty"
        else:
            self.curSpriteName = "instruments_key" if self.color is None else f"instruments_key_{self.color}"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


#
#   A key, whose rust can be removed by a specific mixture of substances
#
class KeyRustyParametric(Object):
    # Constructor
    def __init__(self, world, color=None, isRusted=True):
        # Default sprite name
        self.color = color
        name = "key" if self.color is None else f"{self.color} key"
        defaultSpriteName = "instruments_key" if self.color is None else f"instruments_key_{self.color}"
        Object.__init__(self, world, "key", name, defaultSpriteName)

        self.attributes["isMovable"] = True                       # Can it be moved?
        self.attributes["isPassable"] = True                      # Agen't can't walk over this

        # Container
        self.attributes['isContainer'] = False                    # Is it a container?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Rusted
        self.attributes['isRusted'] = isRusted                    # Is the key rusted?
        self.attributes['rustLevel'] = 3 if isRusted else 0       # Description of the rust (0=none, 1=light, 2=medium, 3=heavy)

        # Key ID
        self.attributes['keyID'] = 1                              # Key ID (1 by default)

        # A chemical dictionary that describes a solution that will remove the rust from this key
        self.attributes['rustRemovalDict'] = {}                   # Dictionary of substances and their proportions in the mixture


    def setKeyID(self, keyID:int):
        self.attributes['keyID'] = keyID

    def setRustRemovalDict(self, rustRemovalDict:dict):
        self.attributes['rustRemovalDict'] = rustRemovalDict

    # Evaluate how close/far the key is from being rust-free.
    def evaluateRustRemovalProgress(self, chemicalDict:dict):
        # Calculate cosine similarity of the two dictionaries (rustRemovalDict and chemicalDict)

        print("ChemicalDict: " + str(chemicalDict))

        # First, get a list of all the keys in the two dictionaries
        allKeys = set(self.attributes['rustRemovalDict'].keys()).union(set(chemicalDict.keys()))
        # Then, create two lists of the values for each key in the dictionaries
        list1 = []
        list2 = []
        for key in allKeys:
            if (key in self.attributes['rustRemovalDict']):
                list1.append(self.attributes['rustRemovalDict'][key])
            else:
                list1.append(0)
            if (key in chemicalDict):
                list2.append(chemicalDict[key])
            else:
                list2.append(0)

        print("Vector (solution): " + str(list1))
        print("Vector (chemical): " + str(list2))

        # Then, normalize each list to unit length
        magnitude1 = math.sqrt(sum([list1[i] ** 2 for i in range(len(list1))]))
        magnitude2 = math.sqrt(sum([list2[i] ** 2 for i in range(len(list2))]))
        list1 = [list1[i] / magnitude1 for i in range(len(list1))]
        list2 = [list2[i] / magnitude2 for i in range(len(list2))]
        # Then, calculate the cosine similarity
        dotProduct = sum([list1[i] * list2[i] for i in range(len(list1))])
        #magnitude1 = math.sqrt(sum([list1[i] ** 2 for i in range(len(list1))]))
        #magnitude2 = math.sqrt(sum([list2[i] ** 2 for i in range(len(list2))]))
        cosineSimilarity = dotProduct #/ (magnitude1 * magnitude2)

        print("### RUST EVALUATION:")
        print("All Keys:" + str(allKeys))
        print("After normalization:")
        print("\tVector (solution): " + str(list1))
        print("\tVector (chemical): " + str(list2))
        print("Dot Product: " + str(dotProduct))



        #print("### RUST COSINE: " + str(cosineSimilarity) + " (dict1: " + str(self.attributes['rustRemovalDict']) + ", dict2: " + str(chemicalDict) + ")")

        return cosineSimilarity



    def tick(self):
        # Call superclass
        Object.tick(self)

        # Clean rust of key (if conditions are met)
        # If the key is in a container, and that container contains SUBSTANCE TODO, and it's rusted, then remove the rust.
        # Check if the key is in a container
        if (self.parentContainer is not None):
            # Get parent container contents
            parentContainerContents = self.parentContainer.contents
            # Check if the parent container contains a specific substance
            containsCleaner = False
            #print("### KEY: Checking for cleaner substance in parent container")
            partialEvidenceLevel = 0

            # Check to make sure there are not more than 1 subtances in the container
            numSubstances = 0
            for obj in parentContainerContents:
                if (obj.type == "substance"):
                    numSubstances += 1

            if (numSubstances > 1):
                #print("## More than 1 substance in container.  Not evaluating rust reduction further, since those subtances may react later.")
                return

            for obj in parentContainerContents:
                #print("## Key: Mixture dictionary: " + str(obj.attributes['mixtureDict']) + " (containsCleaner: " + str(containsCleaner) + ")")
                # Check for "Cleaner"
                if (obj.type == "substance") and (obj.attributes["substanceName"] == "cleaner"):
                    containsCleaner = True
                    break
                # Check for a specific mixture
                if (obj.type == "substance"):
                    cosine = self.evaluateRustRemovalProgress(obj.attributes['mixtureDict'])
                    if (cosine >= 0.99):
                        containsCleaner = True
                        break
                    # Partial evidence -- give some help for hill-climbing. (0=none, 1=light, 2=medium, 3=heavy)
                    elif (cosine >= 0.66):
                        #self.attributes['rustLevel'] = 1
                        partialEvidenceLevel = 1
                    elif (cosine >= 0.33):
                        #self.attributes['rustLevel'] = 2
                        partialEvidenceLevel = 2
                    else:
                        #self.attributes['rustLevel'] = 3
                        partialEvidenceLevel = 3

                    #print("Partial Evidence Level: " + str(partialEvidenceLevel) + " (cosine: " + str(cosine) + ")")

            # If the parent container contains the cleaner substance, then remove the rust from the key
            if (containsCleaner):
                self.attributes['isRusted'] = False
                self.attributes['rustLevel'] = 0    # No more rust
                # Invalidate the sprite
                self.needsSpriteNameUpdate = True

            # Handle partial evidence -- describe the key as less rusty, if certain combinations of substances are found
            #print("### Partial evidence level: " + str(partialEvidenceLevel))
            if (partialEvidenceLevel > 0):
                if (partialEvidenceLevel < self.attributes['rustLevel']):
                    self.attributes['rustLevel'] = partialEvidenceLevel
                    self.needsSpriteNameUpdate = True

        # Update key name based on whether it's rusted or not
        # If the key is rusted, then set it's name to key (rusted)
        if (self.attributes['isRusted']):# and (self.name != "rusted key"):
            if (self.attributes['rustLevel'] == 3):
                self.name = "rusted key (heavily rusted)"
            elif (self.attributes['rustLevel'] == 2):
                self.name = "rusted key (moderately rusted)"
            elif (self.attributes['rustLevel'] == 1):
                self.name = "rusted key (lightly rusted)"

        elif (not self.attributes['isRusted']):
            self.name = "key (no rust)"

        if self.color:
            self.name = f"{self.color} {self.name}"

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        # If rusted, use the rusty sprite
        if (self.attributes['isRusted']):
            self.curSpriteName = "instruments_key_rusty"
        else:
            self.curSpriteName = "instruments_key" if self.color is None else f"instruments_key_{self.color}"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName