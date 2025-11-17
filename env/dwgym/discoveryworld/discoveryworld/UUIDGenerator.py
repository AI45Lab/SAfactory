# UUID Generator

# A class that randomly generates UUIDs, represented as integers. 

# Imports
import random

# Class definition
class UUIDGenerator():
    # Constructor
    def __init__(self, seed:int = 0):
        self.seed = seed
        self.random = random.Random(seed)
        self.existingUUIDs = set()

    # Generate a new UUID
    def generateUUID(self):
        #MAX_UUID = 2**30
        MAX_UUID = 2**16
        uuid = self.random.randint(0, MAX_UUID)
        while uuid in self.existingUUIDs:
            uuid = self.random.randint(0, MAX_UUID)
        self.existingUUIDs.add(uuid)
        return uuid
        