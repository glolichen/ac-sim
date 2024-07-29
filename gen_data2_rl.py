import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import const
import random
import housebuilder
import sys
import stable_baselines3
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import hvac
import argparse

parser = argparse.ArgumentParser(prog="GenData2RL")
parser.add_argument("episodes")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--model")

if __name__ == "__main__":
	args = parser.parse_args()
	if args.output == None:
		args.output = "out.png"
		print("warn: no output passed, default to out.png")
	if args.model == None:
		args.model = "model.pt"
		print("warn: no model passed, default to model.pt")

	env = hvac.Environment()

	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 3])

	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1])

	episode_count = int(args.episodes)
	sim_max = 1440
	num_setpoints = 1

	deviations = {
		"deviation (0)": np.zeros(episode_count),
		"deviation (1)": np.zeros(episode_count),
	}
	cycles = {
		"cycles (ac)": np.zeros(episode_count),
		"cycles (damper) (0)": np.zeros(episode_count),
		"cycles (damper) (1)": np.zeros(episode_count),
	}

	for i in range(episode_count):
		seed_time = time.time()
		random.seed(seed_time)

		house = housebuilder.build_house("2r_simple.json")
		weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

		total_dev0 = 0
		total_dev1 = 0

		damper0_prev = -1
		damper1_prev = -1
		ac_prev = 1000

		damper0_cycle = 0
		damper1_cycle = 0
		ac_cycle = 0

		model = stable_baselines3.DQN.load("./logs/dqn/HVAC-v0_4/rl_model_7200000_steps.zip")

		obs, _ = env.reset(num_setpoints=num_setpoints, length=sim_max, weather_start=weather_start)

		for t in range(sim_max):
			action, _states = model.predict(obs, deterministic=True)
			ac_status, dampers = env.get_action(action)
			obs, _, terminated, _, _ = env.step(action)

			total_dev0 += abs(obs[0] - obs[1]).item()
			total_dev1 += abs(obs[2] - obs[3]).item()

			damper0 = 1 if dampers[0][0] else 0
			damper1 = 1 if dampers[0][1] else 0
			ac_power = house.constants.settings[ac_status]

			if damper0 != damper0_prev:
				damper0_cycle += 1
			if damper1 != damper1_prev:
				damper1_cycle += 1
			if ac_power != ac_prev:
				ac_cycle += 1
			damper0_prev = damper0
			damper1_prev = damper1
			ac_prev = ac_power

		deviations["deviation (0)"][i] = total_dev0 / sim_max
		deviations["deviation (1)"][i] = total_dev1 / sim_max
		cycles["cycles (ac)"][i] = ac_cycle
		cycles["cycles (damper) (0)"][i] = damper0_cycle
		cycles["cycles (damper) (1)"][i] = damper1_cycle

		print(f"{' ' * 20}\r{i + 1}/{episode_count}", end="\r", file=sys.stderr)

	ax0.boxplot(deviations.values())
	ax0.set_xticklabels(deviations.keys())

	ax1.boxplot(cycles.values())
	ax1.set_xticklabels(cycles.keys())

	fig.set_size_inches(12.8, 9.6)
	plt.savefig(args.output, dpi=500, bbox_inches="tight")
