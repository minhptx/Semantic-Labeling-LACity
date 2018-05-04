#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import pickle

import ujson
from pathlib import Path
from typing import Union, Optional


def serialize(obj, fpath: Union[Path, str]):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def serializeCSV(array, fpath: Union[Path, str], delimiter=","):
    with open(fpath, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=delimiter)
        for line in array:
            writer.writerow(line)


def serializeJSON(obj, fpath: Union[Path, str], indent: int=0):
    with open(fpath, 'w') as f:
        # add a special handler to handle case of list of instances of some class
        if type(obj) is list:
            if len(obj) > 0 and hasattr(obj[0], 'to_dict'):
                return ujson.dump((o.to_dict() for o in obj), f, indent=indent)
        elif hasattr(obj, 'to_dict'):
            return ujson.dump(obj.to_dict(), f, indent=indent)

        ujson.dump(obj, f, indent=indent)


def deserialize(fpath: Union[Path, str]):
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def deserializeJSON(fpath: Union[Path, str], Class=None):
    with open(fpath, 'r') as f:
        obj = ujson.load(f)
        if Class is not None:
            if type(obj) is list:
                return [Class.from_dict(o) for o in obj]
            else:
                return Class.from_dict(obj)

        return obj


def deserializeCSV(fpath: Union[Path, str], quotechar: str='"'):
    with open(fpath, "r") as f:
        reader = csv.reader(f, quotechar=quotechar)
        return [row for row in reader]


def serialize2str(obj) -> bytes:
    return pickle.dumps(obj)


def deserialize4str(bstr: bytes):
    return pickle.loads(bstr)
