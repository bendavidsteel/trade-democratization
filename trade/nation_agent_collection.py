class InternationalAgentCollection():
    def __init__(self, num_countries, replay_capacity, num_node_actions, num_global_actions, device):
        self.device = device

        # create agents
        self.agents = []
        for agent_id in range(num_countries):
            new_agent = NationAgent(agent_id, num_countries, replay_capacity, num_node_actions, num_global_actions, device)
            self.agents.append(new_agent)

    def __getitem__(self, idx):
        return self.agents[idx]

    def reset(self, ally_groups, demo_initial):
        new_state_dict = self.get_state()

        # and then apply averaged state dict to agents
        for agent_idx, agent in enumerate(self.agents):
            agent_ally_group = []
            for ally_group in ally_groups:
                if agent_idx in ally_group:
                    agent_ally_group = ally_group
            # reset each individual agent
            agent.reset(new_state_dict, agent_ally_group, demo_initial)

    def get_state(self):
        # get state dict from all agents
        all_agent_states = []
        for agent in self.agents:
            all_agent_states.append(agent.get_state())

        # average them
        new_state_dict = all_agent_states[0]
        for key in new_state_dict:
            for idx in range(1, len(all_agent_states)):
                new_state_dict[key] += all_agent_states[idx][key]
            new_state_dict[key] = new_state_dict[key] / len(all_agent_states)

        return new_state_dict

    def select_actions(self, state, eps_threshold):
        agent_actions = []
        for agent in self.agents:
            action = agent.select_action(state, eps_threshold)
            agent_actions.append(action)
        return agent_actions

    def optimize(self):
        for agent in self.agents:
            agent.optimize()