import importlib.resources
from os.path import join as pjoin

ASSETS_PATH = pjoin(importlib.resources.files("discoveryworld"), "assets")
DATA_PATH = pjoin(importlib.resources.files("discoveryworld"), "data")
