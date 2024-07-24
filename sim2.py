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

	import agents.dumb_agent2
	import agents.pid_agent
	agent = agents.dumb_agent2.agent
	# agent = agents.pid_agent.agent

	fig = plt.figure()
	spec = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[2, 2, 1], hspace=0.25)

	ax0 = fig.add_subplot(spec[0])
	ax1 = fig.add_subplot(spec[1], sharex=ax0, sharey=ax0)
	ax2 = fig.add_subplot(spec[2], sharex=ax1)

	ax00 = ax0
	ax10 = ax1
	ax01 = ax0.twinx()
	ax11 = ax1.twinx()
	ax01.set_ylim([0, 5])
	ax11.set_ylim([0, 5])
	ax2.set_ylim([-1.5, 1.5])

	num_setpoints = 1
	sim_max = 2880
	# sim_max = 1

	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

	xvalues = np.arange(0, sim_max)
	temp0 = np.zeros(sim_max)
	temp1 = np.zeros(sim_max)
	setp0 = np.zeros(sim_max)
	setp1 = np.zeros(sim_max)
	dev0 = np.zeros(sim_max)
	dev1 = np.zeros(sim_max)
	damper0 = np.zeros(sim_max)
	damper1 = np.zeros(sim_max)
	ac_power = np.zeros(sim_max)
	int0 = np.zeros(sim_max)
	int1 = np.zeros(sim_max)
	outside_temp = np.zeros(sim_max)

	total_dev0 = 0
	total_dev1 = 0

	damper0_prev = -1
	damper1_prev = -1
	ac_prev = 1000

	damper0_cycle = 0
	damper1_cycle = 0
	ac_cycle = 0

	change_temp = set([0] + [random.randrange(0, sim_max) for _ in range(num_setpoints - 1)])

	room0: housebuilder.Room = house.get_rooms(0)[0]
	room1: housebuilder.Room = house.get_rooms(0)[1]

	for t in range(sim_max):
		if t in change_temp:
			room0.set_setpoint(random.uniform(20, 28))
			room1.set_setpoint(random.uniform(20, 28))

		temp0[t] = room0.get_temp()
		temp1[t] = room1.get_temp()
		setp0[t] = room0.get_setpoint()
		setp1[t] = room1.get_setpoint()
		int0[t] = house.int_wall_temp[0][0]
		int1[t] = house.int_wall_temp[0][1]
		outside_temp[t] = const.OUTSIDE_TEMP[weather_start + t]

		ac_status, dampers = agent(house, const.OUTSIDE_TEMP[weather_start + t])
		house.step(const.OUTSIDE_TEMP[weather_start + t], ac_status, dampers)

		total_dev0 += abs(room0.get_temp() - room0.get_setpoint())
		total_dev1 += abs(room1.get_temp() - room1.get_setpoint())
		dev0[t] = total_dev0 / (t + 1)
		dev1[t] = total_dev1 / (t + 1)

		damper0[t] = 1 if dampers[0][0] else 0
		damper1[t] = 1 if dampers[0][1] else 0
		ac_power[t] = housebuilder.get_constants().settings[ac_status]

		if damper0[t] != damper0_prev:
			damper0_cycle += 1
		if damper1[t] != damper1_prev:
			damper1_cycle += 1
		if ac_power[t] != ac_prev:
			ac_cycle += 1
		damper0_prev = damper0[t]
		damper1_prev = damper1[t]
		ac_prev = ac_power[t]

		# test[i] = house.int_wall_temp

	ax00.plot(xvalues, temp0, color="red", linewidth=0.1)
	ax00.plot(xvalues, setp0, color="blue", linewidth=0.5)
	ax01.plot(xvalues, dev0, color="purple", linewidth=0.5)
	ax01.plot(xvalues, damper0, color="gray", linewidth=0.2)
	ax00.plot(xvalues, outside_temp, color="green", linewidth=0.1)
	ax00.plot(xvalues, int0, color="orange", linewidth=0.1)

	ax10.plot(xvalues, temp1, color="red", linewidth=0.1)
	ax10.plot(xvalues, setp1, color="blue", linewidth=0.5)
	ax11.plot(xvalues, dev1, color="purple", linewidth=0.5)
	ax11.plot(xvalues, damper1, color="gray", linewidth=0.2)
	ax10.plot(xvalues, outside_temp, color="green", linewidth=0.1)
	ax01.plot(xvalues, int1, color="orange", linewidth=0.1)

	ax2.plot(xvalues, ac_power, linewidth=0.1)

	fig.set_size_inches(9.6, 4.8 / 2 * 5)
	plt.savefig("run.png", dpi=500, bbox_inches="tight")

	print("seed:", seed_time)
	print("dev0", total_dev0 / sim_max)
	print("dev1", total_dev1 / sim_max)
	print("damper0 cycles:", damper0_cycle)
	print("damper1 cycles:", damper1_cycle)
	print("ac cycles:", ac_cycle)
