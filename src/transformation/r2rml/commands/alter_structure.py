#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any

from transformation.models.table_schema import Schema


class ZipAttributesCmd:

    def __init__(self, input_attrs: List[str], output_attr: str):
        self.input_attrs = input_attrs
        self.output_attr = output_attr

        splitted_input_attrs = [self.split_parent(input_attr) for input_attr in input_attrs]
        splitted_output_attr = self.split_parent(output_attr)

        self.scope = splitted_input_attrs[0][0]
        self.attrs = [x[1] for x in splitted_input_attrs]
        self.oattr = splitted_output_attr[1]

        assert all(x[0] == self.scope for x in splitted_input_attrs)
        assert splitted_output_attr[0] == self.scope

        if self.scope == "":
            self.scope = []
        else:
            self.scope = self.scope.split(Schema.PATH_DELIMITER)

    def zip_attributes(self, row: dict):
        self.zip_(row, self.scope)

    def zip_(self, row: dict, parent_paths: List[str]):
        if len(parent_paths) == 0:
            input_cols = [row[attr] for attr in self.attrs]
            n_vals = len(input_cols[0])
            assert all(isinstance(x, list) and len(x) == n_vals for x in input_cols)
            row[self.oattr] = [{attr: val[i] for i, attr in enumerate(self.attrs)} for val in list(zip(*input_cols))]
        else:
            if isinstance(row[parent_paths[0]], list):
                for val in row[parent_paths[0]]:
                    self.zip_(val, parent_paths[1:])
            else:
                self.zip_(row[parent_paths[0]], parent_paths[1:])

    def split_parent(self, attr_path: str) -> Tuple[str, str]:
        idx = attr_path.rfind(Schema.PATH_DELIMITER)
        if idx == -1:
            return "", attr_path
        return attr_path[:idx], attr_path[idx+1:]

    def to_dict(self):
        return {
            "_type_": "ZipAttributes",
            "input_attrs": self.input_attrs,
            "output_attr": self.output_attr
        }

    @staticmethod
    def from_dict(obj: dict):
        return ZipAttributesCmd(obj['input_attrs'], obj['output_attr'])


class UnpackOneElementListCmd:

    def __init__(self, input_attr: str):
        self.input_attr = input_attr
        self.attrs = input_attr.split(Schema.PATH_DELIMITER)

    def unpack(self, row: dict):
        self.unpack_(row, self.attrs)

    def unpack_(self, local_row: dict, attrs: List[str]):
        if len(attrs) == 1:
            if len(local_row[attrs[0]]) == 0:
                local_row[attrs[0]] = None
            else:
                assert len(local_row[attrs[0]]) == 1
                local_row[attrs[0]] = local_row[attrs[0]][0]
        else:
            self.unpack_(local_row[attrs[0]], attrs[1:])

    def to_dict(self):
        return {
            "_type_": "UnpackOneElementList",
            "input_attr": self.input_attr
        }

    @staticmethod
    def from_dict(obj: dict):
        return UnpackOneElementListCmd(obj['input_attr'])
