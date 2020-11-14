import json
import os

import pandas as pd

curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "..", "..", "data")
vdem_path = os.path.join(data_path, "dataset", "raw", "V-Dem-CY-Core-v10.csv")

vdem_df = pd.read_csv(vdem_path)

just_names_df = vdem_df[["country_name","country_text_id"]]
just_names_unique = just_names_df.drop_duplicates()

names = {}
for idx, row in just_names_unique.iterrows():
    names[row['country_text_id']] = row['country_name']

with open(os.path.join(data_path, "country_names.json"), "w") as f:
    f.write(json.dumps(names)) 