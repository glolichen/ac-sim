import imitation.policies.base
import stable_baselines3.common.evaluation
import stable_baselines3.ppo
import numpy as np
import gymnasium as gym
from gym_environment75 import Environment

gym.register(
	id="HVAC-v0",
	entry_point=Environment,
	max_episode_steps=1440,
)
env = gym.make("HVAC-v0")
dagger = imitation.policies.base.FeedForward32Policy.load("dagger_out.zip")

# reward, _ = stable_baselines3.common.evaluation.evaluate_policy(dagger, env, 100)
# print("trained:", np.mean(reward))

model = stable_baselines3.ppo.PPO(imitation.policies.base.FeedForward32Policy, env, verbose=1)
model.policy = dagger

# before_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

model.learn(total_timesteps=1440 * 4000)

# after_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

# print("before:", np.mean(before_reward))
# print("after:", np.mean(after_reward))

model.policy.save("further_train75.zip")
