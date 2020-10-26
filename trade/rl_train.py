GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
NUM_EPISODES = 100
REPLAY_CAPACITY = 20

NUM_COUNTRIES = 20
NUM_YEARS_PER_ROUND = 100

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = NationEnvironment(NUM_COUNTRIES, device)
agents = InternationalAgentCollection(NUM_COUNTRIES, REPLAY_CAPACITY, env.num_foreign_actions, env.num_domestic_actions, device)

for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    env.reset()
    agents.reset(env.clusters, env.norm_initial_demo)

    # reward stats
    reward_mean = 0
    reward_var = 0

    with tqdm.tqdm(range(NUM_YEARS_PER_ROUND)) as years:
        for year in years:
            years.set_postfix(str="Reward Mean: %i, Reward Var: %i" % (reward_mean, reward_var))

            # get state at start of round
            state = env.norm_state

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * i_episode / EPS_DECAY)

            # Select and perform an action
            actions = agents.select_actions(env.norm_state, eps_threshold)
            apply_actions(actions, env)

            # let environment take step
            env.step()

            # Observe new state
            next_state = env.norm_state

            # get the reward
            rewards = env.get_rewards()
            # Store the transition in memory
            for agent_id in range(NUM_COUNTRIES):
                reward = rewards[agent_id]
                action = Action(foreign = actions[agent_id][0],
                                domestic = actions[agent_id][1])
                transition = Transition(state = state,
                                        action = action,
                                        next_state = next_state,
                                        reward = reward)
                agents[agent_id].add_transition(transition)

            reward_mean = torch.mean(rewards)
            reward_var = torch.var(rewards)

            # Perform one step of the optimization (on the target network)
            agents.optimize()