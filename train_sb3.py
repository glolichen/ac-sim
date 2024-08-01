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
	# venv = stable_baselines3.v
	model = stable_baselines3.DQN("MlpPolicy", env, verbose=1,
								gamma=0.9999,
								learning_rate=0.0009145216306356975,
								batch_size=32,
								buffer_size=1000000,
								exploration_final_eps=0.12583623683757106,
								exploration_fraction=0.47604767252903846,
								target_update_interval=1000,
								learning_starts=20000,
								train_freq=16
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
	model.save(args.output)
