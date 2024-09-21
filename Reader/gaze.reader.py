
import collections
import logging
import os
import pickle
import shutil
import traceback as tb
from glob import iglob
from typing import List
import msgpack
import numpy as np


import collections
import msgpack
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pandas as pd

    

class _Empty(object):
    def purge_cache(self):
        pass


class _FrozenDict(dict):
    def __setitem__(self, key, value):
        raise NotImplementedError('Invalid operation')

    def clear(self):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

class Serialized_Dict(object):
    __slots__ = ['_ser_data', '_data']
    cache_len = 100
    _cache_ref = [_Empty()] * cache_len
    MSGPACK_EXT_CODE = 13

    def __init__(self, python_dict=None, msgpack_bytes=None):
        if type(python_dict) is dict:
            self._ser_data = msgpack.packb(python_dict, use_bin_type=True,
                                           default=self.packing_hook)
        elif type(msgpack_bytes) is bytes:
            self._ser_data = msgpack_bytes
        else:
            raise ValueError("Neither mapping nor payload is supplied or wrong format.")
        self._data = None

    def _deser(self):
        if not self._data:
            self._data = msgpack.unpackb(self._ser_data, raw=False, use_list=False,
                                         object_hook=self.unpacking_object_hook,
                                         ext_hook=self.unpacking_ext_hook)
            self._cache_ref.pop(0).purge_cache()
            self._cache_ref.append(self)

    @classmethod
    def unpacking_object_hook(self,obj):
        if type(obj) is dict:
            return _FrozenDict(obj)

    @classmethod
    def packing_hook(self, obj):
        if isinstance(obj, self):
            return msgpack.ExtType(self.MSGPACK_EXT_CODE, obj.serialized)
        raise TypeError("can't serialize {}({})".format(type(obj), repr(obj)))

    @classmethod
    def unpacking_ext_hook(self, code, data):
        if code == self.MSGPACK_EXT_CODE:
            return self(msgpack_bytes=data)
        return msgpack.ExtType(code, data)

    def purge_cache(self):
        self._data = None

    @property
    def serialized(self):
        return self._ser_data

    def __setitem__(self, key, item):
        raise NotImplementedError()

    def __getitem__(self, key):
        self._deser()
        return self._data[key]

    def __repr__(self):
        self._deser()
        return 'Serialized_Dict({})'.format(repr(self._data))

    @property
    def len(self):
        '''Replacement implementation for __len__

        If __len__ is defined numpy will recognize this as nested structure and
        start deserializing everything instead of using this object as it is.
        '''
        self._deser()
        return len(self._data)

    def __delitem__(self, key):
        raise NotImplementedError()

    def get(self,key,default):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        raise NotImplementedError()

    def copy(self):
        self._deser()
        return self._data.copy()

    def has_key(self, k):
        self._deser()
        return k in self._data

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def keys(self):
        self._deser()
        return self._data.keys()

    def values(self):
        self._deser()
        return self._data.values()

    def items(self):
        self._deser()
        return self._data.items()

    def pop(self, *args):
        raise NotImplementedError()

    def __cmp__(self, dict_):
        self._deser()
        return self._data.__cmp__(dict_)

    def __contains__(self, item):
        self._deser()
        return item in self._data

    def __iter__(self):
        self._deser()
        return iter(self._data)

logger = logging.getLogger(__name__)
UnpicklingError = pickle.UnpicklingError

PLData = collections.namedtuple('PLData', ['data', 'timestamps', 'topics'])

    
def serialized_dict_to_dict(ser_dict: Serialized_Dict) -> dict:
    """
    Convert a Serialized_Dict object to a normal Python dictionary.
    """
    result = {}
    for key in ser_dict.keys():
        value = ser_dict[key]
        if isinstance(value, Serialized_Dict):
            # Recursively convert nested Serialized_Dict
            result[key] = serialized_dict_to_dict(value)
        elif isinstance(value, tuple) and hasattr(value, '_asdict'):
            # Convert named tuples to dict
            result[key] = dict(value._asdict())
        else:
            result[key] = value
    return result


def read_gaze(folder_location):
    pupil_data = os.path.join(folder_location, 'gaze.pldata')


    metadata_path = os.path.join(folder_location, "info.player.json")
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    topics = collections.deque()
    with open(pupil_data, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for _, payload in unpacker:
            
            
            unpacked = serialized_dict_to_dict(Serialized_Dict(msgpack_bytes=payload))
            print(unpacked)
            return
            topics.append(unpacked)   
    SYSTEM_START_TIME = metadata["start_time_system_s"]
    PUPIL_LABS_START_TIME = metadata["start_time_synced_s"]
    OFFSET = SYSTEM_START_TIME - PUPIL_LABS_START_TIME
            
    return {"fixations":topics, "offset":OFFSET} 
 
 
read_gaze(os.path.join("./Run3", "000"))