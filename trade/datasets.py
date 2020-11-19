import os

import pandas as pd
import torch
import torch_geometric as geo

class TradeDemoYearByYearDataset(geo.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):

        self.norm_stats = os.path.join(root, "processed", "norm_stats.pt")
        self.node_dict = os.path.join(root, "processed", "node_dict.pt")
        self.year_start = 1901
        self.year_end = 2015

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = []
        # vdem dataset
        files.append("V-Dem-CY-Core-v10.csv")
        # gdp and population data
        files.append("TRADHIST_GDP_POP.xlsx")
        # time invariant bilateral data such as distance, common language
        files.append("TRADHIST_GRAVITY_BILATERAL_TIME_INVARIANT.xlsx")
        # time variant non trade bilateral data such as colony status
        for idx in range(1, 4):
            files.append("TRADHIST_GRAVITY_BILATERAL_TIME_VARIANT_{}.xlsx".format(idx))
        # historical bilateral trade and tariff data
        for idx in range(1, 4):
            files.append("TRADHIST_BITRADE_BITARIFF_{}.xlsx".format(idx))

        return files

    @property
    def processed_file_names(self):
        return ['traddem.pt']

    def process(self):
        # Read data into Data object.
        vdem_nodes = pd.read_csv(os.path.join(self.raw_dir, "V-Dem-CY-Core-v10.csv"))

        tradhist_gdppop = pd.read_excel(os.path.join(self.raw_dir, "TRADHIST_GDP_POP.xlsx"))

        tradhist_timeinvar = pd.read_excel(os.path.join(self.raw_dir, "TRADHIST_GRAVITY_BILATERAL_TIME_INVARIANT.xlsx"))

        tradhist_timevar_frames = []
        for idx in range(1, 4):
            tradhist_timevar_frames.append(pd.read_excel(os.path.join(self.raw_dir, "TRADHIST_GRAVITY_BILATERAL_TIME_VARIANT_{}.xlsx".format(idx))))
        tradhist_timevar = pd.concat(tradhist_timevar_frames)

        tradhist_bitrade_frames = []
        for idx in range(1, 4):
            tradhist_bitrade_frames.append(pd.read_excel(os.path.join(self.raw_dir, "TRADHIST_BITRADE_BITARIFF_{}.xlsx".format(idx))))
        tradhist_bitrade = pd.concat(tradhist_bitrade_frames)

        country_mapping = get_mapping(vdem_nodes, tradhist_timevar)

        num_countries = len(country_mapping)
        num_node_features = 2 + 1 # include GDP and population data, and democracy data from last year
        num_node_targets = 1 # averaged 5 main indicators of democracy from the VDem dataset
        num_edge_features = 7 # Trade flow, current colony relationship, ever a colony, distance, maritime distance, common language, and shared border

        all_years = []
        years_metadata = []
        year_start = self.year_start
        year_end = self.year_end

        for year_idx, year in tqdm.tqdm(enumerate(range(year_start, year_end))):

            year_metadata = {}
            year_metadata["year"] = year

            year_edge_attr = []
            year_edge_index = []

            node_features = []
            node_target = []
            mapping_to_node_indexes = {}

            vdem_this_year = vdem_nodes[(vdem_nodes['year'] == year)]
            vdem_last_year = vdem_nodes[(vdem_nodes['year'] == year - 1)]

            gdppop_year_span = tradhist_gdppop[tradhist_gdppop['year'].isin(list(range(year_start, year + 1)))]

            for country_idx, country_codes in enumerate(country_mapping):
                # check if this year and next is coded in VDem
                vdem_this_year_cty = vdem_this_year[(vdem_this_year['country_text_id'] == country_codes[0])][['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 'v2x_delibdem', 'v2x_egaldem']]
                vdem_last_year_cty = vdem_last_year[(vdem_last_year['country_text_id'] == country_codes[0])][['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 'v2x_delibdem', 'v2x_egaldem']]
                if ((len(vdem_this_year_cty) == 0) | \
                    (len(vdem_last_year_cty) == 0) | \
                    (vdem_this_year_cty.isnull().values.any()) | \
                    (vdem_last_year_cty.isnull().values.any())):
                    continue

                country_features = np.zeros((num_node_features, 1))
                country_targets = np.zeros((num_node_targets, 1))

                # look for last time there was valid value, use 0 otherwise
                gdppop_year_cty = gdppop_year_span[gdppop_year_span['iso'].isin(country_codes)]
                country_features[0] = get_last_valid(gdppop_year_cty, 'GDP')
                country_features[1] = get_last_valid(gdppop_year_cty, 'POP')

                country_features[2] = np.mean(vdem_last_year_cty.values.reshape(-1, 1))
                country_targets[0] = np.mean(vdem_this_year_cty.values.reshape(-1, 1))

                mapping_to_node_indexes[country_idx] = len(node_features)
                node_features.append(np.nan_to_num(country_features))
                node_target.append(np.nan_to_num(country_targets))

            year_metadata["node_mapping"] = mapping_to_node_indexes

            timevar_time_span = tradhist_timevar[tradhist_timevar['year'].isin(list(range(year_start, year + 1)))]
            bitrade_time_span = tradhist_bitrade[tradhist_bitrade['year'].isin(list(range(year_start, year + 1)))]

            # now that all nodes are in this graph have set indexes, we can add the edges with the correct indexes too
            for country_a_idx, node_a_idx in mapping_to_node_indexes.items():

                country_codes_a = country_mapping[country_a_idx]

                time_invar_cty = tradhist_timeinvar[tradhist_timeinvar['iso_d'].isin(country_codes_a)]

                timevar_span_cty = timevar_time_span[timevar_time_span['iso_d'].isin(country_codes_a)]
                bitrade_span_cty = bitrade_time_span[bitrade_time_span['iso_d'].isin(country_codes_a)]

                # we want to normalise imports to a country by the sum of imports for that year
                cty_edges = []
                cty_year_imports = 0

                for country_b_idx, node_b_idx in mapping_to_node_indexes.items():
                    # self links don't really make sense to include in the dataset
                    if country_a_idx == country_b_idx:
                        continue
                    
                    country_codes_b = country_mapping[country_b_idx]

                    bilateral_attr = np.zeros((num_edge_features, 1))
                    # for situations where we have multiple trade links two mapped countries due to how we define a country, we will simply take the last link for now
                    time_invar_attrs = time_invar_cty[time_invar_cty['iso_o'].isin(country_codes_b)]
                    if ((len(time_invar_attrs) == 0) |\
                        (vdem_this_year_cty.isnull().values.all())):
                        continue
                    bilateral_attr[:4] = time_invar_attrs[['Dist_coord', 'Evercol', 'Comlang', 'Contig']].values[-1].reshape(-1, 1)

                    timevar_span_link = timevar_span_cty[timevar_span_cty['iso_o'].isin(country_codes_b)]
                    bitrade_span_link = bitrade_span_cty[bitrade_span_cty['iso_o'].isin(country_codes_b)]

                    bilateral_attr[4] = get_last_valid(timevar_span_link, 'SeaDist_2CST')
                    bilateral_attr[5] = get_last_valid(timevar_span_link, 'Curcol')
                    cty_year_flow = get_last_valid(bitrade_span_link, 'FLOW')
                    bilateral_attr[6] = cty_year_flow
                    cty_year_imports += cty_year_flow

                    edge_index = [node_b_idx, node_a_idx]

                    cty_edges.append(np.nan_to_num(bilateral_attr))
                    year_edge_index.append(edge_index)

                for cty_edge in cty_edges:
                    if cty_year_imports > 0:
                        cty_edge[6] = cty_edge[6] / cty_year_imports
                    year_edge_attr.append(cty_edge)

            year_graph = geo.data.Data(x=torch.tensor(node_features, dtype=torch.float32).view(-1, num_node_features), 
                                       y=torch.tensor(node_target, dtype=torch.float32).view(-1, num_node_targets), 
                                       edge_index=torch.tensor(year_edge_index, dtype=torch.long).T, 
                                       edge_attr=torch.tensor(year_edge_attr, dtype=torch.float32).view(-1, num_edge_features))

            all_years.append(year_graph)
            years_metadata.append(year_metadata)

        # get normalization stats
        # for completely unbiased test we should only get training set stats but will pass on that for this
        stacked_x = torch.cat(list([graph.x for graph in all_years]), 0)
        stacked_y = torch.cat(list([graph.y for graph in all_years]), 0)
        stacked_attrs = torch.cat(list([graph.edge_attr for graph in all_years]), 0)

        x_mean = torch.mean(stacked_x, 0)
        x_std = torch.std(stacked_x, 0)
        y_mean = torch.mean(stacked_y, 0)
        y_std = torch.std(stacked_y, 0)
        attr_mean = torch.mean(stacked_attrs, 0)
        attr_std = torch.std(stacked_attrs, 0)

        for graph in all_years:
            graph.x = (graph.x - x_mean) / x_std
            graph.y = (graph.y - y_mean) / y_std
            graph.edge_attr = (graph.edge_attr - attr_mean) / attr_std

        # save stats for later use
        torch.save({"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "attr_mean": attr_mean, "attr_std": attr_std}, self.norm_stats)
        torch.save(years_metadata, self.node_dict)

        data, slices = self.collate(all_years)
        torch.save((data, slices), self.processed_paths[0])

    def get_norm_stats(self):
        return torch.load(self.norm_stats)