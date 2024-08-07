import imitation.data.wrappers
import imitation.policies.base
import imitation.util.util
import stable_baselines3.common.evaluation
import stable_baselines3.ppo
import numpy as np
import gymnasium as gym
import gym_environment
import argparse

parser = argparse.ArgumentParser(prog="TrainFromDAgger")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--model")
parser.add_argument("-t", "--timesteps")

def main():
	args = parser.parse_args()
	if args.output == None:
		args.output = "models/further_train.zip"
		print("warn: no output passed, default to models/further_train.zip")
	if args.timesteps == None:
		args.timesteps = 500
		print("warn: no timesteps passed, default to 1440 * 500 = 720000")
	if args.model == None:
		args.model = "models/dagger_out.zip"
		print("warn: no model passed, default to models/dagger_out.zip")

	gym.register(
		id="HVAC-v0",
		entry_point=gym_environment.Environment,
		max_episode_steps=1440,
	)
	env = gym.make("HVAC-v0")
	venv = imitation.util.util.make_vec_env(
		"HVAC-v0",
		rng=np.random.default_rng(),
		n_envs=4,
		post_wrappers=[lambda env, _: imitation.data.wrappers.RolloutInfoWrapper(env)],
	)
	dagger = imitation.policies.base.FeedForward32Policy.load(args.model)

	model = stable_baselines3.ppo.PPO(
		imitation.policies.base.FeedForward32Policy, venv, verbose=1,
		batch_size=512, n_steps=2048, gamma=0.9, learning_rate=0.001031942897063114,
		ent_coef=1.1176257099577095e-08, clip_range=0.4, n_epochs=1, gae_lambda=0.8,
		max_grad_norm=0.7, vf_coef=0.5585552871658679
	)
	model.policy = dagger

	before_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

	model.learn(total_timesteps=1440 * int(args.timesteps))

	after_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(model, env, 100)

	print("before:", np.mean(before_reward))
	print("after:", np.mean(after_reward))

	model.policy.save(args.output)

if __name__ == "__main__":
	main()