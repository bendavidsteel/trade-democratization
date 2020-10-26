import json
import os
import random

import polylabel
import tqdm

if __name__ == "__main__":
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "countryBorders.geojson")
    with open(file_path, "r") as f:
        countries_data = json.loads(f.read())

    for country in tqdm.tqdm(countries_data['features']):
        if country['geometry']['type'] == 'MultiPolygon':
            coords = country['geometry']['coordinates'][0]
        else:
            coords = country['geometry']['coordinates']
        centroid = polylabel.polylabel(coords)
        country["centroid"] = centroid

    with open(file_path, "w") as f:
        f.write(json.dumps(countries_data))