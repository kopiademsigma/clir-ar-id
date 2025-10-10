import os
import torch
import ujson
import dataclasses
from typing import Any
from collections import defaultdict
from dataclasses import dataclass, fields
from colbert.utils.utils import timestamp, torch_load_dnn
from dataclasses import MISSING


@dataclass
class DefaultVal:
    val: Any


@dataclass
class CoreConfig:
    def __post_init__(self):
        """
        Source: https://stackoverflow.com/a/58081120/1493011
        """
        self.assigned = {}
        for field in fields(self):
            field_val = getattr(self, field.name)
            if field.default is not MISSING and hasattr(field.default, 'val'):
                setattr(self, field.name, field.default.val)
            if not isinstance(field_val, DefaultVal):
                self.assigned[field.name] = True

    def assign_defaults(self):
        for field in fields(self):
            if field.default is not MISSING and hasattr(field.default, 'val'):
                setattr(self, field.name, field.default.val)
                self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        ignored = set()
        for key, value in kw_args.items():
            self.set(key, value, ignore_unrecognized) or ignored.update({key})
        return ignored
        """
        # TODO: Take a config object, not kw_args.
        for key in config.assigned:
            value = getattr(config, key)
        """

    def set(self, key, value, ignore_unrecognized=False):
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True
        if not ignore_unrecognized:
            raise Exception(f"Unrecognized key `{key}` for {type(self)}")

    def help(self):
        # Convert all values first, then serialize
        config_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # Unwrap DefaultVal objects
            if isinstance(value, DefaultVal):
                value = value.val
            config_dict[field.name] = self._serialize_value(value)
        print(ujson.dumps(config_dict, indent=4))
    
    def _serialize_value(self, obj):
        """Recursively serialize values, handling DefaultVal objects"""
        if isinstance(obj, DefaultVal):
            return self._serialize_value(obj.val)
        elif isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)([self._serialize_value(item) for item in obj])
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Handle custom objects by converting to dict
            return {k: self._serialize_value(v) for k, v in obj.__dict__.items()}
        return obj

    def __export_value(self, v):
        v = v.provenance() if hasattr(v, 'provenance') else v
        if isinstance(v, list) and len(v) > 100:
            v = (f"list with {len(v)} elements starting with...", v[:3])
        if isinstance(v, dict) and len(v) > 100:
            v = (f"dict with {len(v)} keys starting with...", list(v.keys())[:3])
        return v

    def export(self):
        d = dataclasses.asdict(self)
        for k, v in d.items():
            d[k] = self.__export_value(v)
        return d