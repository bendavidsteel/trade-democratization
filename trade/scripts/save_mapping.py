import json
import os

import pandas as pd

from utils import get_mapping

if __name__ == "__main__":
    
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    dataset = os.path.join(root, "dataset")

    vdem_nodes = pd.read_csv(os.path.join(dataset, "raw", "V-Dem-CY-Core-v10.csv"))

    tradhist_timevar_frames = []
    for idx in range(1, 4):
        tradhist_timevar_frames.append(pd.read_excel(os.path.join(dataset, "raw", "TRADHIST_GRAVITY_BILATERAL_TIME_VARIANT_{}.xlsx".format(idx))))
    tradhist_timevar = pd.concat(tradhist_timevar_frames)

    country_mapping = get_mapping(vdem_nodes, tradhist_timevar)

    file_path = os.path.join(root, "data", "country_mapping.json")
    with open(file_path, "w") as f:
        f.write(json.dumps(country_mapping))