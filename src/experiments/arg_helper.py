#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from semantic_modeling.data_io import get_semantic_models


def str2bool(v):
    assert v.lower() in {"true", "false"}
    return v.lower() == "true"


def parse_kfold(dataset, kfold_arg):
    def get_sm_ids_by_name_range(start_name, end_name, sm_ids):
        start_idx, end_idx = None, None
        for i, sid in enumerate(sm_ids):
            if sid.startswith(start_name):
                assert start_idx is None
                start_idx = i
            if sid.startswith(end_name):
                assert end_idx is None
                end_idx = i

        assert start_idx <= end_idx
        return sm_ids[start_idx:end_idx + 1]  # inclusive

    # support some shorthand like: {train_sm_ids: ["s08-s21"], test_sm_ids: ["s01-s07", "s22-s28"]}, inclusive
    kfold_arg = ujson.loads(kfold_arg)
    sm_ids = sorted([sm.id for sm in get_semantic_models(dataset)])
    train_sm_ids = []
    for shorthand in kfold_arg['train_sm_ids']:
        start, end = shorthand.split("-")
        train_sm_ids += get_sm_ids_by_name_range(start, end, sm_ids)
    test_sm_ids = []
    for shorthand in kfold_arg['test_sm_ids']:
        start, end = shorthand.split("-")
        test_sm_ids += get_sm_ids_by_name_range(start, end, sm_ids)
    kfold_arg['train_sm_ids'] = train_sm_ids
    kfold_arg['test_sm_ids'] = test_sm_ids

    return kfold_arg
