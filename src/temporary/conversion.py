import csv
import os

import yaml

for file_name in os.listdir("data/la_city/sources"):
    print(file_name)
    semantic_list = []

    with open("data/la_city/sources/%s" % file_name, "r") as f:
        reader = csv.reader(f)

        column_names = next(reader)

        semantic_types = next(reader)

        print(column_names)
        for column_name, semantic_type in zip(column_names, semantic_types):
            semantic_obj = {"_type_": "SetSemanticType", "domain": "LACity",
                            "node_id": "LACity1",
                            "type": semantic_type,
                            "input_attr_path": column_name}
            semantic_list.append(semantic_obj)

    with open("data/la_city/models-y2rml/%s-model.yml" % os.path.splitext(file_name)[0], "w") as f:
        yaml.dump({"commands": semantic_list}, f, default_flow_style=False)
