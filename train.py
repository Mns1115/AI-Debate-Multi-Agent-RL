import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from debate_env import DebateEnv

# Register the custom environment
def env_creator(config):
    return DebateEnv()

register_env("DebateEnv", env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Configure PPO algorithm
config = (
    PPOConfig()
    .environment("DebateEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=2)
    .training(model={"fcnet_hiddens": [256, 256]}, train_batch_size=4000)
)

# Create trainer and start training
trainer = config.build()
for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: Reward {result['episode_reward_mean']}")

# Save the trained model
trainer.save("debate_model")
