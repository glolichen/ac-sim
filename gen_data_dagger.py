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
import gym_environment
import argparse
import imitation
import imitation.policies
import imitation.policies.base
from typing import Union, Dict

parser = argparse.ArgumentParser(prog="GenData2RL")
parser.add_argument("episodes")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--model")

def main():
	args = parser.parse_args()
	if args.output == None:
		args.output = "out.png"
		print("warn: no output passed, default to out.png")
	if args.model == None:
		args.model = "dagger_out.zip"
		print("warn: no model passed, default to dagger_out.zip")

	num_rooms = 5

	env = gym_environment.Environment()

	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[num_rooms, num_rooms + 1])

	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1])
	
	ax0.set_ylim([0, 3])
	ax1.set_ylim([0, 500])

	episode_count = int(args.episodes)
	sim_max = 1440
	num_setpoints = 1

	deviations = {k: v for (k, v) in zip(
		[f"deviation ({i})" for i in range(num_rooms)],
		[np.zeros(episode_count) for _ in range(num_rooms)]
	)}

	cycles = {k: v for (k, v) in zip(
		[f"cycles (damper) ({i})" for i in range(num_rooms)],
		[np.zeros(episode_count) for _ in range(num_rooms)]
	)}
	cycles["cycles (ac)"] = np.zeros(episode_count)

	model = imitation.policies.base.FeedForward32Policy.load(args.model)

	for ie in range(episode_count):
		house = housebuilder.build_house("2r_simple.json")
		weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

		total_devs = np.zeros(num_rooms)
		damper_prevs = np.full(num_rooms, -1)
		damper_cycles = np.zeros(num_rooms)
		ac_prev = 1000
		ac_cycle = 0
		obs, _ = env.reset(num_setpoints=num_setpoints, length=sim_max, weather_start=weather_start)

		for t in range(sim_max):
			action, _ = model.predict(obs, deterministic=True)
			ac_status, dampers = env.get_action(action)
			obs, _, terminated, _, _ = env.step(action)

			for i in range(num_rooms):
				total_devs[i] += abs(obs[i * 2] - obs[i * 2 + 1]).item()

			if ac_status != ac_prev:
				ac_cycle += 1
			ac_prev = ac_status

			for i in range(num_rooms):
				total_devs[i] += abs(obs[i * 2] - obs[i * 2 + 1]).item()
				if dampers[0][i] != damper_prevs[i]:
					damper_cycles[i] += 1
				damper_prevs[i] = dampers[0][i]

		for i in range(num_rooms):
			deviations[f"deviation ({i})"][ie] = total_devs[i] / sim_max
			cycles[f"cycles (damper) ({i})"][ie] = damper_cycles[i]
		cycles["cycles (ac)"][ie] = ac_cycle

		print(f"{' ' * 20}\r{ie + 1}/{episode_count}", end="\r", file=sys.stderr)

	ax0.boxplot(deviations.values())
	ax0.set_xticklabels(deviations.keys())

	ax1.boxplot(cycles.values())
	ax1.set_xticklabels(cycles.keys())

	fig.set_size_inches(5.4 * num_rooms, 9.6)
	plt.savefig(args.output, dpi=1000, bbox_inches="tight")

if __name__ == "__main__":
	main()
