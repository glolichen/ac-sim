import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import const
import random
import housebuilder
import sys
import time
import argparse
from agents.generalized_dumb_agent import agent

parser = argparse.ArgumentParser(prog="GenData2Dumb")
parser.add_argument("episodes")
parser.add_argument("-o", "--output")

def main():
	args = parser.parse_args()
	if args.output == None:
		args.output = "out/out.png"
		print("warn: no output passed, default to out/out.png")

	num_rooms = 2

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

	for ie in range(episode_count):
		seed_time = time.time()
		random.seed(seed_time)

		house = housebuilder.build_house("2r_simple.json")
		
		weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

		total_devs = np.zeros(num_rooms)
		damper_prevs = np.full(num_rooms, -1)
		damper_cycles = np.zeros(num_rooms)
		ac_prev = 1000
		ac_cycle = 0

		change_temp = set([0] + [random.randrange(0, sim_max) for _ in range(num_setpoints - 1)])

		rooms = house.get_rooms(0)

		for t in range(sim_max):
			if t in change_temp:
				for room in rooms:
					room.set_setpoint(random.uniform(20, 28))

			ac_status, dampers_a = agent(house, const.OUTSIDE_TEMP[weather_start + t])
			house.step(const.OUTSIDE_TEMP[weather_start + t], ac_status, dampers_a)
			
			for i in range(num_rooms):
				total_devs[i] += abs(rooms[i].get_temp() - rooms[i].get_setpoint())
				if dampers_a[0][i] != damper_prevs[i]:
					damper_cycles[i] += 1
				damper_prevs[i] = dampers_a[0][i]

			if ac_status != ac_prev:
				ac_cycle += 1
			ac_prev = ac_status

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
