import stable_baselines3
import gym_environment
import argparse

parser = argparse.ArgumentParser(prog="TrainRL")
parser.add_argument("-o", "--output")
parser.add_argument("-t", "--timesteps")

if __name__ == "__main__":
	args = parser.parse_args()
	if args.output == None:
		args.output = "model"
		print("warn: no output passed, default to model.zip")
	if args.timesteps == None:
		args.timesteps = 500
		print("warn: no model passed, default to 1440 * 500 = 720000")

	if args.output.endswith(".zip"):
		args.output = args.output.rstrip(".zip")

	env = gym_environment.Environment("2r_simple.json")
	model = stable_baselines3.A2C("MlpPolicy", env, verbose=2,
								learning_rate = 0.001,
								n_steps = 10,
								gamma = 0.98,
								gae_lambda = 0.95,
								ent_coef = 0.01,
								vf_coef = 0.4,
								max_grad_norm = 0.7,
								rms_prop_eps = 0.0001

								# learning_rate = 0.00,
								# n_steps = 5,
								# gamma = 0.99,
								# gae_lambda = 1,
								# ent_coef = 0,
								# vf_coef = 0.5,
								# max_grad_norm = 0.5,
								# rms_prop_eps = 0.00001
	)
	model.learn(total_timesteps=int(args.timesteps) * 1440, log_interval=1440)
	model.save("ppo_house")
