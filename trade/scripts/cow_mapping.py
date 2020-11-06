import json
import os

import pandas as pd

if __name__ == "__main__":
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_path, "..", "..", "data")
    mappings_path = os.path.join(data_path, "country_mapping.json")
    
    with open(mappings_path, "r") as f:
        curr_mappings = json.loads(f.read())

    cow_mappings = []
    for mapping in curr_mappings:
        cow_map = { "demtrad": mapping, "cow": [] }
        cow_mappings.append(cow_map)

    cow_mapping_path = os.path.join(data_path, "cow_mapping.json")
    with open(cow_mapping_path, "w") as f:
        f.write(json.dumps(cow_mappings))

    cow_codes_path = os.path.join(data_path, "dataset", "raw", "COW_country_codes.csv")
    cow_codes_df = pd.read_csv(cow_codes_path)

    cow_country_codes = list(cow_codes_df['StateAbb'].unique())
    other_3_codes = [codes[0] for codes in curr_mappings]

    cow_code_ex = []

    for cow_code in cow_country_codes:
        match = False
        for map_codes in curr_mappings:
            if cow_code in map_codes:
                match = True
                other_3_codes.remove(map_codes[0])

        if not match:
            cow_code_ex.append(cow_code)

    print("COW codes not in mapping: " + str(cow_code_ex))
    print("Mapping codes not in COW codes: " + str(other_3_codes))