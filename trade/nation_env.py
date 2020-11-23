import itertools
import os
import random

import torch
import torch_geometric as geo

import nets as net

class NationEnvironment():
    def __init__(self, num_countries, device):
        curr_path = os.path.dirname(__file__)
        root = os.path.join(curr_path, "..", "data")
        self.norm_stats = torch.load(os.path.join(root, "dataset", "processed", "norm_stats.pt"))
        best_model = torch.load(os.path.join(root, "models", 'best_model_recurrent.pkl'))

        self.num_countries = num_countries
        self.device = device

        num_node_features = 2
        num_edge_features = 7
        num_output_features = 1
        self.env_model = net.RecurGraphNet(num_node_features, num_edge_features, num_output_features).to(device)
        self.env_model.load_state_dict(best_model)
        self.reset()

        self.num_foreign_actions = 5
        self.num_domestic_actions = 4
        
    def reset(self):
        self.initial_demo = torch.rand(self.num_countries, 1, dtype=torch.float32)
        self.norm_initial_demo = (self.initial_demo - self.norm_stats["y_mean"]) / self.norm_stats["y_std"]

        # start with up to 1 thousand gdp and 1 million pop
        gdp = 1000000000 * torch.rand(self.num_countries, 1, dtype=torch.float32)
        pop = 1000000 * torch.rand(self.num_countries, 1, dtype=torch.float32)
        self.node_features = torch.cat([gdp,
                                        pop], dim=1)

        # establish country ally clusters
        self.clusters = []
        cluster_edges = []
        num_clusters = self.num_countries // 10
        for cluster_idx in range(num_clusters):
            cluster = random.sample(list(range(self.num_countries)), random.randint(2, self.num_countries // 5))
            self.clusters.append(cluster)
            for edge in list(itertools.permutations(cluster, 2)):
                cluster_edges.append(edge)

        # starting with number of links on average anywhere between 1 and 5
        num_edges = (self.num_countries * random.randint(1, 5)) + len(cluster_edges)
        self.edge_indexes = torch.randint(self.num_countries, (2, num_edges), dtype=torch.long)

        for idx in range(len(cluster_edges)):
            self.edge_indexes[0, idx] = cluster_edges[idx][0]
            self.edge_indexes[1, idx] = cluster_edges[idx][1]

        # ensure no self links
        for idx in range(self.edge_indexes.shape[1]):
            if self.edge_indexes[0,idx] == self.edge_indexes[1,idx]:
                if self.edge_indexes[1,idx] == self.num_countries - 1:
                    self.edge_indexes[1,idx] -= 1
                else:
                    self.edge_indexes[1,idx] += 1

        # ever col -> curr col
        #          -> common language
        ever_col = (torch.rand(num_edges, 1) > 0.98)
        curr_col = ((torch.rand(num_edges, 1) > 0.5) * ever_col)
        com_lang = ((torch.rand(num_edges, 1) > 0.9) | ((torch.rand(num_edges, 1) > 0.5) * ever_col))
        # distance -> distance by sea
        #          -> shared borders
        #          -> trade
        coor_dis = 15000 * torch.rand(num_edges, 1, dtype=torch.float32)
        sea_dist = coor_dis * ((2.5 * torch.rand(num_edges, 1, dtype=torch.float32)) + 1)
        trad_imp = coor_dis * 10000 * torch.rand(num_edges, 1, dtype=torch.float32)
        shar_bor = (((coor_dis < 1000) * (torch.rand(num_edges, 1) > 0.5)) | ((coor_dis < 2000) * (torch.rand(num_edges, 1) > 0.7)) | ((coor_dis < 5000) * (torch.rand(num_edges, 1) > 0.9))).float()
        # order of edge features is distance, ever a colony, common language, shared borders, distance by sea, current colony, imports
        self.edge_features = torch.cat([coor_dis.float(),
                                        ever_col.float(),
                                        com_lang.float(),
                                        shar_bor.float(),
                                        sea_dist.float(),
                                        curr_col.float(),
                                        trad_imp.float()], dim=1)
        
        self.env_model.reset(self.norm_initial_demo)

        self.create_normed_state()

        
    def establish_trade(self, agent_id, target_id):
        # ensure no self links
        if agent_id == target_id:
            return

        # origin country index comes first
        trade_link = torch.tensor([target_id, agent_id]).view(2,1)
        for idx in range(self.edge_indexes.shape[1]):
            if ((self.edge_indexes[0,idx] == trade_link[0,0]) and (self.edge_indexes[1,idx] == trade_link[1,0])):
                # trade link already established
                return

        # create features for new link
        ever_col = 0
        curr_col = 0
        com_lang = random.random() > 0.9
        coor_dis = 15000 * random.random()
        sea_dist = coor_dis * ((2.5 * random.random()) + 1)
        trad_imp = coor_dis * 10000 * random.random()
        shar_bor = ((coor_dis < 1000) * (random.random() > 0.5)) | ((coor_dis < 2000) * (random.random() > 0.7)) | ((coor_dis < 5000) * (random.random() > 0.9))
        new_features = torch.tensor([coor_dis,
                                     ever_col,
                                     com_lang,
                                     shar_bor,
                                     sea_dist,
                                     curr_col,
                                     trad_imp]).view(1, 7)

        self.edge_features = torch.cat((self.edge_features, new_features), dim=0)
        self.edge_indexes = torch.cat((self.edge_indexes, trade_link), dim=1)

    def increase_imports(self, agent_id, target_id):
        self.scale_imports(agent_id, target_id, 1.05, 1.3)

    def decrease_imports(self, agent_id, target_id):
        self.scale_imports(agent_id, target_id, 0.7, 0.95)

    def scale_imports(self, agent_id, target_id, lower_bound, upper_bound):
        link_idx = -1
        for idx in range(self.edge_indexes.shape[1]):
            if (self.edge_indexes[0,idx] == target_id) and (self.edge_indexes[1,idx] == agent_id):
                link_idx = idx
                break

        if link_idx == -1:
            return
        
        self.edge_features[idx, 6] = self.edge_features[idx, 6] * random.uniform(lower_bound, upper_bound)

    def colonize(self, agent_id, target_id):
        # check if there is a link with this country
        # and ensure no other country has already colonized the target country
        link_idx = -1
        already_colonised = False
        for idx in range(self.edge_indexes.shape[1]):
            if (self.edge_indexes[0,idx] == target_id) and (self.edge_indexes[1,idx] == agent_id):
                link_idx = idx

            if self.edge_indexes[0,idx] == target_id:
                if self.edge_features[idx, 5] == 1:
                    already_colonised = True

        if link_idx == -1 | already_colonised:
            return

        # colonizing country needs to be bigger
        if (self.node_features[agent_id, 0] > 1.2 * self.node_features[target_id, 0]) and \
           (self.node_features[agent_id, 1] > 1.1 * self.node_features[target_id, 1]):
            self.edge_features[link_idx, 5] = 1
            self.edge_features[link_idx, 1] = 1

    def decolonize(self, agent_id, target_id):
        # check if there is a link with this country
        link_idx = -1
        for idx in range(self.edge_indexes.shape[1]):
            if (self.edge_indexes[0,idx] == target_id) and (self.edge_indexes[1,idx] == agent_id):
                link_idx = idx
                break

        if link_idx == -1:
            return

        if self.edge_features[link_idx, 5] == 1:
            self.edge_features[link_idx, 5] = 0

    def increase_gdp(self, agent_id):
        self.node_features[agent_id, 0] += 0.2 * self.node_features[agent_id, 0] * (random.random() + 0.5)

    def decrease_gdp(self, agent_id):
        self.node_features[agent_id, 0] -= 0.2 * self.node_features[agent_id, 0] * (random.random() + 0.5)

    def increase_pop(self, agent_id):
        self.node_features[agent_id, 1] += 0.2 * self.node_features[agent_id, 1] * (random.random() + 0.5)

    def decrease_pop(self, agent_id):
        self.node_features[agent_id, 1] -= 0.2 * self.node_features[agent_id, 1] * (random.random() + 0.5)

    def step(self):
        # gdp and pop fluctuations
        self.node_features[:, 0] += 0.05 * self.node_features[:, 0] * (torch.rand(self.num_countries, dtype=torch.float32) - 0.5)
        self.node_features[:, 1] += 0.05 * self.node_features[:, 1] * (torch.rand(self.num_countries, dtype=torch.float32) - 0.5)

        # colonized countries can flip to having a common language
        one_feat_shape = self.edge_features[:, 2].shape
        self.edge_features[:, 2] = torch.min(torch.ones(one_feat_shape), self.edge_features[:, 2] + (self.edge_features[:, 5] * (torch.rand(one_feat_shape) > 0.9)))

        # sea distance can shorten
        self.edge_features[:, 4] = torch.min(self.edge_features[:, 0] * 1.5, self.edge_features[:, 4] * torch.max(torch.ones(one_feat_shape), 0.8 + torch.rand(one_feat_shape) * 20))

        data = geo.data.Data(x=self.node_features, edge_index=self.edge_indexes, edge_attr=self.edge_features)
        self.node_demo = self.env_model(data)

        self.create_normed_state()
        
    def create_normed_state(self):
        self.norm_state = geo.data.Data(x = (self.node_features.clone() - self.norm_stats["x_mean"][:2]) / self.norm_stats["x_std"][:2],
                                        edge_index = self.edge_indexes,
                                        edge_attr = (self.edge_features.clone() - self.norm_stats["attr_mean"]) / self.norm_stats["attr_std"])

    def get_rewards(self):
        rewards = torch.zeros(self.num_countries)
        for country_idx in range(self.num_countries):
            demo = self.node_demo[country_idx, 0]
            
            # add reward for own country 
            rewards[country_idx] += 4 * demo

            # add rewards for ally countries
            for cluster in self.clusters:
                if country_idx in cluster:
                    for cluster_country_idx in cluster:
                        if cluster_country_idx != country_idx:
                            rewards[cluster_country_idx] += 2 * demo
                
        return rewards

