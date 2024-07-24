import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import const
import random
import housebuilder
import sys
import time

if __name__ == "__main__":
	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 3])

	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1])

	episode_count = int(sys.argv[1])
	sim_max = 1440
	num_setpoints = 1

	deviations = {
		"deviation (0)": np.zeros(episode_count),
	}
	cycles = {
		"cycles (ac)": np.zeros(episode_count),
	}

	for i in range(episode_count):
		seed_time = time.time()
		random.seed(seed_time)

		house = housebuilder.build_house("1r_simple.json")

		import agents.dumb_agent1
		agent = agents.dumb_agent1.agent

		weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

		total_dev0 = 0
		ac_prev = 1000
		ac_cycle = 0

		change_temp = set([0] + [random.randrange(0, sim_max) for _ in range(num_setpoints - 1)])

		room0: housebuilder.Room = house.get_rooms(0)[0]

		for t in range(sim_max):
			if t in change_temp:
				room0.set_setpoint(random.uniform(20, 28))
			ac_status, dampers = agent(house, const.OUTSIDE_TEMP[weather_start + t])
			house.step(const.OUTSIDE_TEMP[weather_start + t], ac_status, dampers)

			total_dev0 += abs(room0.get_temp() - room0.get_setpoint())
			ac_power = housebuilder.get_constants().settings[ac_status]

			if ac_power != ac_prev:
				ac_cycle += 1
			ac_prev = ac_power

		deviations["deviation (0)"][i] = total_dev0 / sim_max
		cycles["cycles (ac)"][i] = ac_cycle

		if total_dev0 / sim_max > 1.5:
			print(f"warn: seed {seed_time} dev0 {total_dev0 / sim_max}")

		print(f"{' ' * 20}\r{i + 1}/{episode_count}", end="\r", file=sys.stderr)

	ax0.boxplot(deviations.values())
	ax0.set_xticklabels(deviations.keys())

	ax1.boxplot(cycles.values())
	ax1.set_xticklabels(cycles.keys())

	fig.set_size_inches(12.8, 4.8)
	plt.savefig("out.png", dpi=500, bbox_inches="tight")
