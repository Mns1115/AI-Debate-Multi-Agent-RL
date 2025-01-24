from pettingzoo.utils.env import ParallelEnv
import numpy as np
import gymnasium

class DebateEnv(ParallelEnv):
    def __init__(self):
        self.possible_agents = ["agent_1", "agent_2"]
        self.observation_space = {agent: gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32) for agent in self.possible_agents}
        self.action_space = {agent: gymnasium.spaces.Discrete(3) for agent in self.possible_agents}

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: np.random.rand(5) for agent in self.agents}
        return self.observations

    def step(self, actions):
        rewards = {}
        for agent in self.agents:
            if actions[agent] == 0:  # Logical argument
                rewards[agent] = 1
            elif actions[agent] == 1:  # Persuasive argument
                rewards[agent] = 2
            else:  # Invalid argument
                rewards[agent] = -1

        self.rewards.update(rewards)
        done = {agent: np.random.rand() > 0.95 for agent in self.agents}  # Random end condition
        return self.observations, rewards, done, {}

    def render(self):
        print(f"Agents: {self.agents}, Observations: {self.observations}")
