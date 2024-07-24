import housebuilder
import const
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time

if __name__ == "__main__":
	seed_time = time.time() if len(sys.argv) <= 1 else float(sys.argv[-1])
	random.seed(seed_time)

	house = housebuilder.build_house("1r_simple.json")

	import agents.dumb_agent1
	agent = agents.dumb_agent1.agent

	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=1, ncols=1)

	ax0 = fig.add_subplot(spec[0])
	ax00 = ax0
	ax01 = ax0.twinx()

	num_setpoints = 5
	sim_max = 2880

	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

	xvalues = np.arange(0, sim_max)
	temp0 = np.zeros(sim_max)
	setp0 = np.zeros(sim_max)
	dev0 = np.zeros(sim_max)
	ac_power = np.zeros(sim_max)
	outside_temp = np.zeros(sim_max)

	total_dev0 = 0
	damper0_prev = -1
	ac_prev = 1000
	ac_cycle = 0

	change_temp = set([0] + [random.randrange(0, sim_max) for _ in range(num_setpoints - 1)])

	room0: housebuilder.Room = house.get_rooms(0)[0]

	for t in range(sim_max):
		if t in change_temp:
			room0.set_setpoint(random.uniform(20, 28))

		temp0[t] = room0.get_temp()
		setp0[t] = room0.get_setpoint()
		outside_temp[t] = const.OUTSIDE_TEMP[weather_start + t]

		ac_status, dampers = agent(house, const.OUTSIDE_TEMP[weather_start + t])
		house.step(const.OUTSIDE_TEMP[weather_start + t], ac_status, dampers)

		total_dev0 += abs(room0.get_temp() - room0.get_setpoint())
		dev0[t] = total_dev0 / (t + 1)
		ac_power[t] = housebuilder.get_constants().settings[ac_status]

		if ac_power[t] != ac_prev:
			ac_cycle += 1
		ac_prev = ac_power[t]

		# test[i] = house.int_wall_temp

	ax00.plot(xvalues, temp0, color="red", linewidth=0.1)
	ax00.plot(xvalues, setp0, color="blue", linewidth=0.5)
	ax01.plot(xvalues, dev0, color="purple", linewidth=0.5)
	ax00.plot(xvalues, outside_temp, color="green", linewidth=0.1)
	ax01.plot(xvalues, ac_power, linewidth=0.1)

	fig.set_size_inches(9.6, 4.8)
	plt.savefig("run.png", dpi=500, bbox_inches="tight")

	print("seed:", seed_time)
	print("dev0", total_dev0 / sim_max)
	print("ac cycles:", ac_cycle)
