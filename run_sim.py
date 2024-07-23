import const
import housebuilder
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time

if __name__ == "__main__":
	seed_time = time.time() if len(sys.argv) <= 1 else float(sys.argv[-1])
	random.seed(seed_time)

	house = housebuilder.build_house("2r_simple.json")

	import agents.very_dumb_agent
	agent = agents.very_dumb_agent.agent

	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 2], hspace=0.25)

	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1], sharex=ax0, sharey=ax0)

	sim_max = 610

	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

	xvalues = np.arange(0, sim_max)
	temp0 = np.zeros(sim_max)
	temp1 = np.zeros(sim_max)
	int0 = np.zeros(sim_max)
	int1 = np.zeros(sim_max)
	outside_temp = np.zeros(sim_max)

	room0 = house.get_rooms(0)[0]
	room1 = house.get_rooms(0)[1]

	zero = housebuilder.get_constants().settings.index(0)

	for i in range(sim_max):
		action = zero
		if i == 600:
			action = housebuilder.get_constants().settings.index(1)
		temp0[i] = room0.get_temp()
		temp1[i] = room1.get_temp()
		int0[i] = house.int_wall_temp[0][0]
		int1[i] = house.int_wall_temp[0][1]
		outside_temp[i] = const.OUTSIDE_TEMP[weather_start + i]

		print(i, end=" ")

		house.step(const.OUTSIDE_TEMP[weather_start + i], action, [[False, False]])

	ax0.plot(xvalues, temp0, color="red", linewidth=0.1)
	ax0.plot(xvalues, int0, color="orange", linewidth=0.1)
	ax0.plot(xvalues, outside_temp, color="green", linewidth=0.1)

	ax1.plot(xvalues, temp1, color="red", linewidth=0.1)
	ax1.plot(xvalues, int1, color="orange", linewidth=0.1)
	ax1.plot(xvalues, outside_temp, color="green", linewidth=0.1)

	fig.set_size_inches(9.6, 4.8 * 2)
	plt.savefig("run_sim.png", dpi=500, bbox_inches="tight")

	print("seed:", seed_time)
