# JSONEncoder.py

import json
from enum import Enum

from discoveryworld.objects import Object
from discoveryworld.Layer import Layer


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Object):
            return {"objUUID": obj.uuid}
        if isinstance(obj, Layer):
            return str(obj)
        if isinstance(obj, Enum):
            return str(obj)

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
