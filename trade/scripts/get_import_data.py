import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from trade import utils

if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

    mapping_path = os.path.join(data_path, "country_mapping.json")
    with open(mapping_path, "r") as f:
        country_mapping = json.loads(f.read())

    borders_path = os.path.join(data_path, "capitals.geojson")
    with open(borders_path, "r") as f:
        borders_data = json.loads(f.read())

    cty_lookup = {}

    for country_idx in range(len(country_mapping)):
        this_border_idx = -1
        for border_idx in range(len(borders_data["features"])):
            border_data = borders_data["features"][border_idx]
            if border_data["properties"]["iso3"] == country_mapping[country_idx][0]:
                this_border_idx = border_idx

        """ if this_border_idx == -1:
            cty_code = cty_mapping[country_idx][0]
            breakpoint() """

        cty_lookup[country_idx] = this_border_idx

    
    raw_dir = os.path.join(data_path, "dataset", "raw")

    tradhist_bitrade_frames = []
    for idx in range(1, 4):
        tradhist_bitrade_frames.append(pd.read_excel(os.path.join(raw_dir, "TRADHIST_BITRADE_BITARIFF_{}.xlsx".format(idx))))
    tradhist_bitrade = pd.concat(tradhist_bitrade_frames)

    num_countries = len(country_mapping)
    years_metadata = []
    year_start = 1901
    year_end = 2015

    for year_idx, year in tqdm.tqdm(enumerate(range(year_start, year_end))):

        year_metadata = {}
        year_metadata["year"] = year

        year_edges = []

        bitrade_time_span = tradhist_bitrade[tradhist_bitrade['year'].isin(list(range(year_start, year + 1)))]

        # now that all nodes are in this graph have set indexes, we can add the edges with the correct indexes too
        for country_a_idx, country_codes_a in enumerate(country_mapping):

            bitrade_span_cty = bitrade_time_span[bitrade_time_span['iso_d'].isin(country_codes_a)]

            for country_b_idx, country_codes_b in enumerate(country_mapping):
                # self links don't really make sense to include in the dataset
                if country_a_idx == country_b_idx:
                    continue

                bitrade_span_link = bitrade_span_cty[bitrade_span_cty['iso_o'].isin(country_codes_b)]

                flow = utils.get_last_valid(bitrade_span_link, 'FLOW')

                if flow <= 0:
                    continue

                edge = [cty_lookup[country_b_idx], cty_lookup[country_a_idx], flow]

                year_edges.append(edge)

        year_metadata["trade_data"] = year_edges

        years_metadata.append(year_metadata)

    visual_data_path = os.path.join(data_path, "visual_trade_path.json")
    with open(visual_data_path, "w") as f:
        f.write(json.dumps(years_metadata))