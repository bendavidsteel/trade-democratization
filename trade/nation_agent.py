import torch
import torch_geometric as geo

class NationAgent():
    def __init__(self, agent_id, num_countries, replay_capacity, num_node_actions, num_global_actions, device):
        # more node features because we will add indicator of self country and ally countries
        num_node_features, num_edge_features = 4, 7

        # create two DQNs for stable learning
        self.policy_net = RecurGraphAgent(num_node_features, num_edge_features, num_node_actions, num_global_actions).to(device)
        self.target_net = RecurGraphAgent(num_node_features, num_edge_features, num_node_actions, num_global_actions).to(device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

        self.memory = ReplayMemory(replay_capacity)

        # ensure they match
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.agent_id = agent_id
        self.num_countries = num_countries
        self.num_node_actions = num_node_actions
        self.num_global_actions = num_global_actions
        self.device = device


    def reset(self, state_dict, ally_countries, demo_initial):
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

        self.policy_net.reset(demo_initial)
        self.target_net.reset(demo_initial)

        # create node data with features for self and ally countries
        # using -0.1 and 0.9 as approximation of normalization
        self.node_features = -0.1 * torch.ones((self.num_countries, 4), dtype=torch.float32)
        self.node_features[self.agent_id, 2] = 0.9
        for ally_idx in ally_countries:
            self.node_features[ally_idx, 3] = 0.9

        batch = torch.zeros(self.num_countries, dtype=torch.long, device=self.device)
        self.memory.reset(demo_initial, batch)

    def get_state(self):
        return self.policy_net.state_dict()

    def select_action(self, env_state, eps_threshold):
        # add in country specific state
        self.node_features[:, :2] = env_state.x[:,:2]

        state = geo.data.Data(x=self.node_features,
                              edge_index=env_state.edge_index.clone(),
                              edge_attr=env_state.edge_attr.clone())
        
        state.batch = torch.zeros(self.num_countries, dtype=torch.long, device=self.device)

        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                foreign_output, domestic_output = self.policy_net(state)
                foreign_action = torch.argmax(foreign_output)
                domestic_action = torch.argmax(domestic_output)
                return foreign_action, domestic_action
        else:
            return torch.tensor(random.randrange(self.num_node_actions), device=self.device, dtype=torch.long), torch.tensor(random.randrange(self.num_global_actions), device=self.device, dtype=torch.long)

    def add_transition(self, transition):
        # add in country specific state
        self.node_features[:, :2] = transition.state.x[:,:2]
        transition.state.x = self.node_features

        self.node_features[:, :2] = transition.next_state.x[:,:2]
        transition.next_state.x = self.node_features

        self.memory.push(transition)

    def optimize(self):
        # single transition because i haven't worked out how to make batches work with net yet
        transition = self.memory.sample()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        foreign_output, domestic_output = self.policy_net(transition.state, step=False)
        state_action_values = foreign_output[transition.action.foreign] + domestic_output[transition.action.domestic]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # this environment never technically ends, so we shouldn't expect the agent to predict a final step of rewards
        
        foreign_output, domestic_output = self.target_net(transition.next_state, step=False)
        next_state_values = foreign_output.max().detach() + domestic_output.max().detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + transition.reward

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()