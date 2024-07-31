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

	env = gym_environment.Environment()
	model = stable_baselines3.DQN("MlpPolicy", env, verbose=1,
								learning_rate = 1e-4,
								gamma = 0.99,
								tau = 0.001,
								exploration_initial_eps = 0.99,
								exploration_final_eps = 0.001,
								exploration_fraction = 0.1,
								batch_size = 188

								# learning_rate = 0.00,
								# n_steps = 5,
								# gamma = 0.99,
								# gae_lambda = 1,
								# ent_coef = 0,
								# vf_coef = 0.5,
								# max_grad_norm = 0.5,
								# rms_prop_eps = 0.00001
	)
	model.learn(total_timesteps=int(args.timesteps) * 1440, log_interval=10)
	model.save("dqn_house")
