# ScienceHelpers.py
# A collection of helper functions that are useful for calculating e.g. various physical science quantities


#
#   Radiation level
#

# Window size is the size of the window in which to calculate the radiation level. It must be an odd number (e.g. 1, 3, 5, etc.)
# Anything "held up to" the detector is considered to be at distance 1.  Things in the same tile are considered distance 2, things 1 tile away are considered distance 3, etc.
# 'excludeObject' is nominally a specific object being measured by the radiation sensor (i.e. at distance 1), that we want to exclude from this background rate calculation.
def getRadiationLevelAroundLocation(world, gridX, gridY, excludeObject=None, windowSize=5):
    # Get half the window size
    halfWindowSize = int((windowSize-1) / 2)

    totalRadiationLevel = 0

    for x in range(gridX-halfWindowSize, gridX+halfWindowSize+1):
        for y in range(gridY-halfWindowSize, gridY+halfWindowSize+1):
            #print("Examining radiation level at (" + str(x) + ", " + str(y) + ")")
            if (world.isWithinBounds(x, y)):
                # Get the radiation level at the current location
                radiationLevel = getRadiationLevelAtLocation(world, x, y, excludeObject=excludeObject)
                #print("\t Radiation level at (" + str(x) + ", " + str(y) + ") is " + str(radiationLevel) + " uSv/h")
                # Get the distance between (gridX, gridY) and (x, y)
                distance = (((gridX-x)**2 + (gridY-y)**2)**0.5) + 2     # Plus 2 so things at the same tile will be considered distance 2.
                #print("\t Distance: " + str(distance))
                # Inverse square law -- divide radiation level by the square of the distance
                radiationLevel /= distance**2
                #print("\t After inverse square law, radiation level is " + str(radiationLevel) + " uSv/h")
                # Add the radiation level to the total
                totalRadiationLevel += radiationLevel
                #print("\t Running total: " + str(totalRadiationLevel) + " uSv/h")

    #print("total: " + str(totalRadiationLevel) + " uSv/h")
    return totalRadiationLevel


# Returns the total radiation level at a given location
def getRadiationLevelAtLocation(world, gridX, gridY, excludeObject=None):
    # Get all objects at the current location
    allObjectsAndParts = world.getObjectsAt(gridX, gridY, respectContainerStatus=False, includeParts=True)

    totalRadiationLevel = 0
    for obj in allObjectsAndParts:
        if (excludeObject == None) or (obj.uuid != excludeObject.uuid):     # Make sure we're not including an object marked for exclusion.
            for material in obj.attributes["materials"]:
                if ("radiationusvh" in material):
                    totalRadiationLevel += material["radiationusvh"]

    return totalRadiationLevel




#
#   NPK Content of soil
#
def getNPKContent(obj):
        # Get the patient object, and all its parts
        # def getAllContainedObjectsAndParts(self, includeContents=True, includeParts=True):
        patientObjAndParts = obj.getAllContainedObjectsAndParts(includeContents=True, includeParts=True)
        # Collect the materials of the object and its parts
        patientMaterials = []
        for patientObjOrPart in patientObjAndParts:
            if ("materials" in patientObjOrPart.attributes):
                patientMaterials.extend(patientObjOrPart.attributes["materials"])

        # Get the nitrogen, phosphorus, and potassium content of the materials
        nitrogen = []
        phosphorus = []
        potassium = []
        for patientMaterial in patientMaterials:
            if ("nitrogen" in patientMaterial):
                nitrogen.append(patientMaterial["nitrogen"])
            if ("phosphorus" in patientMaterial):
                phosphorus.append(patientMaterial["phosphorus"])
            if ("potassium" in patientMaterial):
                potassium.append(patientMaterial["potassium"])

        # Sum the nitrogen, phosphorus, and potassium content of the materials
        totalNitrogen = sum(nitrogen)
        totalPhosphorus = sum(phosphorus)
        totalPotassium = sum(potassium)

        # Pack
        npkContent = {"nitrogen": totalNitrogen, "phosphorus": totalPhosphorus, "potassium": totalPotassium}
        return npkContent


#
#   PH of objects
#
def getAveragePH(obj):
    # Just get average PH
    phs = []

    # Get the patient object, and all its parts
    # def getAllContainedObjectsAndParts(self, includeContents=True, includeParts=True):
    patientObjAndParts = obj.getAllContainedObjectsAndParts(includeContents=True, includeParts=True)
    # Collect the materials of the object and its parts
    patientMaterials = []
    for patientObjOrPart in patientObjAndParts:
        if ("materials" in patientObjOrPart.attributes):
            patientMaterials.extend(patientObjOrPart.attributes["materials"])

    # Get the PH of the materials
    for patientMaterial in patientMaterials:
        if ("ph" in patientMaterial):
            phs.append(patientMaterial["ph"])

    if (len(phs) == 0):
        return None

    # Get the average PH
    totalPH = sum(phs)
    averagePH = totalPH / len(phs)

    return averagePH





#
#   Heat or cool all the objects subjected to a heat/cooling source
#

def heatObjects(objList:list, finalTemperature:float, maxTemperatureChange:float=25):
    # For every object in the object list
    for obj in objList:
        # Get the current temperature of the object
        currentTemperature = obj.attributes["temperatureC"]
        # Get the temperature difference between the current temperature and the final temperature
        temperatureDifference = finalTemperature - currentTemperature

        # If there's a positive difference, start heating
        if (temperatureDifference > 0):
            # If the temperature difference is greater than the maximum temperature change, only change the temperature by the maximum temperature change
            if (temperatureDifference > maxTemperatureChange):
                obj.attributes["temperatureC"] += maxTemperatureChange   # Heat the object by the maximum temperature change this step
            else:
                obj.attributes["temperatureC"] += temperatureDifference  # Heat the object by the temperature difference this step

def coolObjects(objList:list, finalTemperature:float, maxTemperatureChange:float=25):
    # For every object in the object list
    for obj in objList:
        # Get the current temperature of the object
        currentTemperature = obj.attributes["temperatureC"]
        # Get the temperature difference between the current temperature and the final temperature
        temperatureDifference = currentTemperature - finalTemperature

        # If there's a positive difference, start heating
        if (temperatureDifference > 0):
            # If the temperature difference is greater than the maximum temperature change, only change the temperature by the maximum temperature change
            if (temperatureDifference > maxTemperatureChange):
                obj.attributes["temperatureC"] -= maxTemperatureChange   # Cool the object by the maximum temperature change this step
            else:
                obj.attributes["temperatureC"] -= temperatureDifference  # Cool the object by the temperature difference this step


#
#   Living Things -- set dead if outside of temperature range
#
def livingTemperatureRangeCheck(obj):
    # Check to see if the temperature goes out of range of the living things tolerance.  If so, set the living thing to dead.
    # Can only perform this check if the living thing has a material specified.
    if (len(obj.attributes["materials"]) > 0) and (obj.attributes['isLiving'] == True):
        curTemperature = obj.attributes['temperatureC']
        # Should only be one material
        material = obj.attributes["materials"][0]
        if (material['isLiving'] == True):
            if (curTemperature < material['livingMinTemp']) or (curTemperature > material['livingMaxTemp']):
                # Outside of living range
                obj.attributes['isLiving'] = False
                print("### " + obj.name + " is dead, due to being out of temperature range.  Temperature is " + str(curTemperature) + " C.  Living range is " + str(material['livingMinTemp']) + " C to " + str(material['livingMaxTemp']) + " C.  ###")
