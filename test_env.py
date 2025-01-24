from debate_env import DebateEnv

# Create the environment
env = DebateEnv()

# Reset the environment
obs = env.reset()
print("Initial Observations:", obs)

# Simulate a step in the environment
actions = {agent: env.action_space[agent].sample() for agent in env.possible_agents}
obs, rewards, dones, _ = env.step(actions)

print("Observations:", obs)
print("Rewards:", rewards)
print("Dones:", dones)
