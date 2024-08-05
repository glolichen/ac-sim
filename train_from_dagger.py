import imitation.policies.base
import stable_baselines3.common.evaluation
import stable_baselines3.ppo
import numpy as np
import gymnasium as gym
import gym_environment
import argparse

parser = argparse.ArgumentParser(prog="TrainFromDAgger")
parser.add_argument("-o", "--output")
parser.add_argument("-t", "--timesteps")

def main():
	args = parser.parse_args()
	if args.output == None:
		args.output = "models/model.zip"
		print("warn: no output passed, default to model.zip")
	if args.timesteps == None:
		args.timesteps = 500
		print("warn: no model passed, default to 1440 * 500 = 720000")

	gym.register(
		id="HVAC-v0",
		entry_point=gym_environment.Environment,
		max_episode_steps=1440,
	)
	env = gym.make("HVAC-v0")
	dagger = imitation.policies.base.FeedForward32Policy.load("dagger_out.zip")

	model = stable_baselines3.ppo.PPO(imitation.policies.base.FeedForward32Policy, env, verbose=1)
	model.policy = dagger

	before_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

	model.learn(total_timesteps=1440 * int(args.timesteps))

	after_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

	print("before:", np.mean(before_reward))
	print("after:", np.mean(after_reward))

	model.policy.save(args.output)
