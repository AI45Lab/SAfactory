# ReactorLab.py
from functools import reduce
import math
from discoveryworld.ActionSuccess import ActionSuccess
from discoveryworld.objects.Object import Object


class QuantumCrystal(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "quantum crystal", "quantum crystal", defaultSpriteName = "instruments_quantum_crystal")

        # Resonance Frequency of the crystal (a set property for a given crystal)
        self.attributes['resonanceFreq'] = 5000                    # The resonance frequency of the crystal

        # Quantities that the crystal depends on
        self.attributes['density'] = 1.0                          # The density of the crystal (in g/cm^3)
        self.attributes['temperatureC'] = 25.0                    # The temperature of the crystal (in degrees C)
        self.attributes['quantumSize'] = 2.0                      # The quantum size of the crystal (in nm)
        # Add a faux material, with a given radiation and spectrum
        fauxMaterial = {}
        fauxMaterial['radiationusvh'] = 10.0                           # The radiation of the crystal (in mSv)
        fauxMaterial['spectrum'] = [1.0, 2.0, 3.0, 4.0, 5.0]           # The spectrum of the crystal (in nm)
        fauxMaterial['microscopeDesc'] = "The quantum gap of this crystal appears to be " + str(self.attributes['quantumSize']) + " nm"  # The description of the crystal under a microscope
        self.attributes['materials'].append(fauxMaterial)


    def tick(self):
        # Call superclass
        Object.tick(self)


class GeneratorSideLeft(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "generator (left side)", "generator", defaultSpriteName = "instruments_generator_left")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class GeneratorSideRight(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "generator (right side)", "generator", defaultSpriteName = "instruments_generator_right")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

    def tick(self):
        # Call superclass
        Object.tick(self)


class GeneratorCenter(Object):
    # Constructor
    def __init__(self, world):
        # Default sprite name
        Object.__init__(self, world, "generator core", "generator", defaultSpriteName = "instruments_generator_center_off")

        # Default attributes
        self.attributes["isMovable"] = False                       # Can it be moved?
        self.attributes["isPassable"] = False                      # Agen't can't walk over this

        # Device (is activable)
        self.attributes['isActivatable'] = False                   # Is this a device? (more specifically, can it be activated/deactivated?)
        self.attributes['isActivated'] = True                      # Is this device currently activated?

        # Material
        self.attributes["manualMaterialNames"] = ["Metal"]

        # Object-specific properties
        self.attributes['linkedObjectsActivationState'] = []       # List of objects that are linked to this object
        self.linkedObjectsActivationState = []                     # List of objects that are linked to this object (the object references)

        # Also store the 'percentage activation'
        self.percentageActivation = 0.0

    def addLinkedObjectsActivationState(self, objs:list):
        for obj in objs:
            self.attributes['linkedObjectsActivationState'].append(obj.uuid)



    def tick(self):
        # First, check if the linked object references are set.  If not, then set them.
        # This is so the object serializes correctly.  The `attributes` must all be serializable, and can't be object references.
        # We could look this up each tick, but it would be expensive.
        if (len(self.attributes['linkedObjectsActivationState']) != len(self.linkedObjectsActivationState)):
            self.linkedObjectsActivationState = []
            for worldObj in self.world.getAllWorldObjects():
                if (worldObj.uuid in self.attributes['linkedObjectsActivationState']):
                    self.linkedObjectsActivationState.append(worldObj)

        # Check if all the objects that it's paired with are activated
        numActivated = 0
        lastPercentageActivation = self.percentageActivation
        if (len(self.linkedObjectsActivationState) > 0):
            for obj in self.linkedObjectsActivationState:
                if (not obj.attributes['isActivated']):
                    allActivted = False
                else:
                    numActivated += 1
            # Normalize numActivated
            self.percentageActivation = numActivated / len(self.linkedObjectsActivationState)

        # Check if there was a change
        if (lastPercentageActivation != self.percentageActivation):
            self.needsSpriteNameUpdate = True

        # Update 'isActivated' attribute based on the state of the linked objects
        if (self.percentageActivation >= 0.99):
            self.attributes['isActivated'] = True
        else:
            self.attributes['isActivated'] = False

        # Change name based on percentage activated
        self.name = "generator core (" + str(int(self.percentageActivation * 100)) + "% activated)"


        # Call superclass
        Object.tick(self)


    # Sprite
    # Updates the current sprite name based on the current state of the object
    def inferSpriteName(self, force:bool=False):
        if (not self.needsSpriteNameUpdate and not force):
            # No need to update the sprite name
            return

        self.curSpriteName = self.defaultSpriteName
        if (self.attributes['isActivated']):
            self.curSpriteName = "instruments_generator_center_on"
        elif (self.percentageActivation < 0.25):
            self.curSpriteName = "instruments_generator_center_off"
        elif (self.percentageActivation < 0.50):
            self.curSpriteName = "instruments_generator_center_25"
        elif (self.percentageActivation < 0.75):
            self.curSpriteName = "instruments_generator_center_50"
        elif (self.percentageActivation < 1.0):
            self.curSpriteName = "instruments_generator_center_75"

        else:
            # Should never see this one, unless the activation is huge
            self.curSpriteName = "instruments_generator_center_off"

        # This will be the next last sprite name (when we flip the backbuffer)
        self.tempLastSpriteName = self.curSpriteName
