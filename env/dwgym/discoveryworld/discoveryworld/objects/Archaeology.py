from discoveryworld.ActionSuccess import ActionSuccess, MessageImportance
from discoveryworld.objects.Object import Object
    #


class AncientArtifact(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "ancient artifact", "ancient artifact", defaultSpriteName = "placeholder_jar_empty")

    def tick(self):
        # Call superclass
        Object.tick(self)

        #self.name = "ancient artifact (" + str(self.attributes['radiocarbonAge']) + " years old)"


class ArtifactBronzeChisel(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "ancient bronze chisel", "ancient bronze chisel", defaultSpriteName = "instruments_bronze_chisel")

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class ArtifactIronTongs(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "ancient iron tongs", "ancient iron tongs", defaultSpriteName = "instruments_iron_tongs")

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class ArtifactStoneHammer(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "ancient stone hammer", "ancient stone hammer", defaultSpriteName = "instruments_stone_hammer")

        # Material
        self.attributes["manualMaterialNames"] = ["Rock"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class RadioCarbonMeter(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "radiocarbon meter", "radiocarbon meter", defaultSpriteName = "instruments_radiocarbon_meter")

        # Default attributes

        self.attributes['isUsable'] = True                       # Can this device be used with another object? (e.g. specifically through the 'use' action)

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        # Use this object on the patient object
        useDescriptionStr = "You use the radiocarbon meter to investigate the " + patientObj.name + ".\n"

        # Check for a "radiocarbon_age" attribute in the patient object
        radiocarbonAge = -1
        if ("radiocarbonAge" in patientObj.attributes):
            radiocarbonAge = patientObj.attributes["radiocarbonAge"]

        # If the radiocarbon age is less than or equal to zero, then the results are inconclusive
        if (radiocarbonAge <= 0):
            useDescriptionStr += "The results are inconclusive.\n"
            return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

        # If the radiocarbon age is greater than zero, then report the age
        useDescriptionStr += "The radiocarbon meter estimates an age of " + str(radiocarbonAge) + " years.\n"
        return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

    #
    #   Tick
    #
    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName


class RadioisotopeMeter(Object):
    # Constructor
    def __init__(self, world):
        Object.__init__(self, world, "radioisotope meter", "radioisotope meter", defaultSpriteName = "instruments_generic_meter")

        # Default attributes

        self.attributes['isUsable'] = True                       # Can this device be used with another object? (e.g. specifically through the 'use' action)

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        pass


    #
    #   Actions (use with)
    #
    def actionUseWith(self, patientObj):
        ### TODO: CURRENTLY JUST COPIED/PASTED FROM THE SPECTROMETER.

        # Use this object on the patient object
        useDescriptionStr = "You use the radioisotope meter to view the " + patientObj.name + ".\n"

        # seedMediumArtifact.attributes["radioisotopeValues"]
        # Check for a "radioisotopeValues" attribute in the patient object
        if ("radioisotopeValues" not in patientObj.attributes):
            useDescriptionStr += "The results are inconclusive.\n"
            return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

        # If the list is empty, then the results are inconclusive
        if (len(patientObj.attributes["radioisotopeValues"]) == 0):
            useDescriptionStr += "The results are inconclusive.\n"
            return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

        # If there are values, then report the values
        useDescriptionStr += "The results are as follows:\n"
        for i in range(len(patientObj.attributes["radioisotopeValues"])):
            useDescriptionStr += "- Radioisotope " + str(i+1) + ": " + "{:.3f}".format(patientObj.attributes["radioisotopeValues"][i]) + "\n"

        return ActionSuccess(True, useDescriptionStr, importance=MessageImportance.HIGH)

    #
    #   Tick
    #
    def tick(self):
        # TODO: Invalidate sprite name if this or neighbouring walls change
        if (False):
            self.needsSpriteNameUpdate = True

        # TODO

        # Call superclass
        Object.tick(self)

    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = self.defaultSpriteName

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName
