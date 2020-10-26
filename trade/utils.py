import collections
import random

import numpy as np
import torch_geometric as geo
import torch

def get_mapping(vdem_nodes, tradhist_timevar):   

    vdem_country_codes = list(vdem_nodes['country_text_id'].unique())
    tradhist_country_codes = list(tradhist_timevar['iso_o'].unique())
    shared_codes = [code for code in vdem_country_codes if code in tradhist_country_codes]

    mapped_codes = [['RUS', 'USSR'], ['YEM', 'ADEN'], ['CAF', 'AOFAEF', 'FRAAEF'], ['TCD', 'AOFAEF', 'FRAAEF'], ['COD', 'AOFAEF', 'FRAAEF'], ['HRV', 'AUTHUN', 'YUG'], ['SVK', 'CZSK', 'AUTHUN'], 
                ['SVN', 'AUTHUN', 'YUG'], ['UKR', 'AUTHUN', 'USSR'], ['ALB', 'AUTHUN'], ['BIH', 'AUTHUN', 'YUG'], ['MNE', 'AUTHUN', 'YUG'], ['CAN', 'CANPRINCED', 'CANQBCONT', 'NFLD'], 
                ['CZE', 'CZSK', 'AUTHUN'], ['DDR', 'EDEU'], ['MYS', 'FEDMYS', 'UNFEDMYS', 'GBRBORNEO'], ['BFA', 'FRAAOF'], ['GNQ', 'FRAAOF'], ['LUX', 'ZOLL'], 
                ['ZZB', 'ZANZ', 'GBRAFRI'], ['ZAF', 'ZAFTRA', 'ZAFORA', 'ZAFNAT', 'ZAPCAF', 'GBRAFRI'], ['MKD', 'YUG'], ['SRB', 'YUG'], ['POL', 'USSR'], ['COM', 'MYT'], ['ROU', 'ROM'], 
                ['MWI', 'RHOD', 'GBRAFRI'], ['ZMB', 'RHOD', 'GBRAFRI'], ['ZWE', 'RHOD', 'GBRAFRI'], ['SGP', 'STRAITS'], ['DEU', 'WDEU'], ['SML', 'GBRSOM', 'ITASOM'], ['GBR', 'ULSTER'], 
                ['RWA', 'RWABDI'], ['SOM', 'ITASOM'], ['MAR', 'MARESP'], ['FRA', 'OLDENB'], ['DNK', 'SCHLES'], ['LBN', 'SYRLBN', 'OTTO'], ['SYR', 'SYRLBN'], ['CYP', 'OTTO', 'GBRMEDI'], 
                ['TUR', 'OTTO'], ['STP', 'PRTAFRI'], ['AGO', 'PRTAFRI'], ['MOZ', 'PRTAFRI'], ['GNB', 'PRTWAFRI'], ['KHM', 'INDOCHI'], ['LAO', 'INDOCHI'], ['VNM', 'INDOCHI'], 
                ['ERI', 'ITAEAFRI', 'GBRAFRI'], ['TTO', 'GBRWINDIES'], ['SLE', 'GBRWAFRI'], ['GMB', 'GBRWAFRI'], ['TGO', 'GBRWAFRI'], ['EGY', 'OTTO'],
                ['PNG', 'GBRPAPUA'], ['MLT', 'GBRMEDI'], ['BGD', 'GBRIND'], ['BTN', 'GBRIND'], ['IND', 'GBRIND'], ['MDV', 'GBRIND'], ['NPL', 'GBRIND'], ['PAK', 'GBRIND'], 
                ['LKA', 'GBRIND'], ['CMR', 'GBRAFRI', 'FRAAFRI'], ['KEN', 'GBRAFRI'], ['SYC', 'GBRAFRI'], ['SDN', 'GBRAFRI'], ['UGA', 'GBRAFRI'], ['LSO', 'GBRAFRI'], 
                ['SWZ', 'GBRAFRI']]

    # validate my matches
    code_count = {}
    for codes in mapped_codes:
        matched_to_vdem = 0
        for code in codes:
            if len(code) == 3:
                if code in vdem_country_codes:
                    if code in code_count:
                        code_count[code] += 1
                    else:
                        code_count[code] = 1
                    matched_to_vdem += 1

        if matched_to_vdem == 0:
            raise ValueError("{} country code set matched to no VDem node".format(codes))
        elif matched_to_vdem > 1:
            raise ValueError("{} country code set matched to more than one VDem node".format(codes))

        if codes[0] not in vdem_country_codes:
            raise ValueError("VDem code should be first in list {}.".format(codes))

    for code in code_count:
        if code_count[code] != 1:
            raise ValueError("VDem code {} matched to more than one country code set".format(code))

    for code in shared_codes:
        if code not in code_count:
            mapped_codes.append([code])

    return mapped_codes


def get_last_valid(df, col):
    valid_rows = df[df[col].isnull() == False]
    if (len(valid_rows) > 0):
        return np.nan_to_num(valid_rows.sort_values('year')[col].values)[-1]
    else:
        return 0


Action = collections.namedtuple('Action', ('foreign', 'domestic'))
NumpyData = collections.namedtuple('NumpyData', ('x', 'edge_index', 'edge_attr'))
State = collections.namedtuple('State', ('initial', 'sequence', 'batch'))
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def to_numpy_data(data):
    return NumpyData(x = data.x.detach().numpy(),
                     edge_index = data.edge_index.detach().numpy(),
                     edge_attr = data.edge_attr.detach().numpy())
    
def data_from_numpy(data):
    return geo.data.Data(x = torch.from_numpy(data.x),
                         edge_index = torch.from_numpy(data.edge_index),
                         edge_attr = torch.from_numpy(data.edge_attr))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity

    def reset(self, initial, batch):
        self.initial = initial.numpy()
        self.batch = batch.numpy()
        self.states = []
        self.actions = []
        self.rewards = []

    def push(self, transition):
        """Saves a transition."""
        if not self.states:
            self.states.append(to_numpy_data(transition.state))
        self.states.append(to_numpy_data(transition.next_state))
        self.actions.append(Action(foreign = transition.action.foreign.detach().numpy(),
                                   domestic = transition.action.domestic.detach().numpy()))
        self.rewards.append(transition.reward.detach().numpy())

    def sample(self):
        sample_idx = random.randint(max(0, len(self.states) - 2 - self.capacity), len(self.states) - 2)
        return Transition(state = State(initial = torch.from_numpy(self.initial), sequence = [data_from_numpy(state) for state in self.states[:sample_idx + 1]], batch = torch.from_numpy(self.batch)),
                          action = Action(foreign = torch.from_numpy(self.actions[sample_idx].foreign), domestic = torch.from_numpy(self.actions[sample_idx].domestic)),
                          reward = torch.from_numpy(self.rewards[sample_idx]),
                          next_state = State(initial = torch.from_numpy(self.initial), sequence = [data_from_numpy(state) for state in self.states[:sample_idx + 2]], batch = torch.from_numpy(self.batch)))

    def __len__(self):
        return len(self.memory)                 