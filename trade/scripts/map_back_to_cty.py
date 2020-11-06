import os
import json

if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

    mapping_path = os.path.join(data_path, "country_mapping.json")
    with open(mapping_path, "r") as f:
        country_mapping = json.loads(f.read())

    borders_path = os.path.join(data_path, "capitals.geojson")
    with open(borders_path, "r") as f:
        borders_data = json.loads(f.read())

    cty_lookup = {}

    for city_idx in range(len(borders_data["features"])):

        city_code = borders_data["features"][city_idx]["properties"]["iso3"]

        this_cty_idx = -1
        for cty_idx in range(len(country_mapping)):
            cty_code = country_mapping[cty_idx][0]
            if cty_code == city_code:
                this_cty_idx = cty_idx

        cty_lookup[city_idx] = this_cty_idx

    visual_data_path = os.path.join(data_path, "visual_trade_data.json")
    with open(visual_data_path, "r") as f:
        trade_data = json.loads(f.read())

    new_trade_data = []

    for year_trade in trade_data:

        new_year_trade = {}
        new_year_trade["year"] = year_trade["year"]

        new_trades = []
        for trade in year_trade["trade_data"]:

            if (trade[0] == -1) or (trade[1] == -1):
                continue

            new_trade = []

            new_trade.append(cty_lookup[trade[0]])
            new_trade.append(cty_lookup[trade[1]])
            new_trade.append(trade[2])

            new_trades.append(new_trade)

        new_year_trade["trade_data"] = new_trades

        new_trade_data.append(new_year_trade)

    visual_data_path = os.path.join(data_path, "visual_trade_path.json")
    with open(visual_data_path, "w") as f:
        f.write(json.dumps(new_trade_data))