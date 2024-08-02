import imitation.policies.base
import stable_baselines3.common.evaluation
import numpy as np
import gymnasium as gym
import gym_environment

gym.register(
	id="HVAC-v0",
	entry_point=gym_environment.Environment,
	max_episode_steps=1440,
)
env = gym.make("HVAC-v0")
dagger = imitation.policies.base.FeedForward32Policy.load("dagger_out2.zip")

reward, _ = stable_baselines3.common.evaluation.evaluate_policy(dagger, env, 100)
print("expert:", np.mean(reward))

