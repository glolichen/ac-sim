import const
import housebuilder
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time
import argparse
from typing import List
# from agents.dumb_agent2 import agent
from agents.generalized_dumb_agent import agent

parser = argparse.ArgumentParser(prog="Simulator")
parser.add_argument("-o", "--output")
parser.add_argument("-t", "--time")
parser.add_argument("-h", "--house")
parser.add_argument("-a", "--agent")

def main():
	global num
	args = parser.parse_args()
	if args.output == None:
		args.output = "out.png"
		print("warn: no output passed, default to out.png")
	if args.time == None:
		args.time = 1440
		print("warn: no time passed, default to 1440")
	if args.house == None:
		print("error: no house entered!")
		sys.exit(1)
	if args.agent == None:
		print("error: no agent entered!")
		sys.exit(1)

	house = housebuilder.build_house(args.house)

	num_rooms = len(house.get_rooms(0))

	fig = plt.figure()
	spec = gridspec.GridSpec(
		nrows=num_rooms + 1,
		ncols=1,
		height_ratios=[2 for _ in range(num_rooms)] + [1]
	)

	ax0 = fig.add_subplot(spec[0])
	axes: List[plt.Axes] = [ax0]
	for i in range(1, num_rooms):
		axes.append(fig.add_subplot(spec[i], sharex=ax0, sharey=ax0))
	status_ax = fig.add_subplot(spec[-1], sharex=ax0)
	status_ax.set_ylim([-1.5, 1.5])
	
	ax_left: List[plt.Axes]  = []
	ax_right: List[plt.Axes] = []
	for ax in axes:
		ax_left.append(ax)
		ax_right.append(ax.twinx())
		ax_right[-1].set_ylim([0, 5])

	num_setpoints = 1
	sim_max = int(args.time)

	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

	xvalues = np.arange(0, sim_max)
	
	temps = [np.zeros(sim_max) for _ in range(num_rooms)]
	setps = [np.zeros(sim_max) for _ in range(num_rooms)]
	devs = [np.zeros(sim_max) for _ in range(num_rooms)]
	dampers = [np.zeros(sim_max) for _ in range(num_rooms)]
	ac_power = np.zeros(sim_max)
	outside_temp = np.zeros(sim_max)

	total_devs = [0 for _ in range(num_rooms)]
	damper_prevs = [-1 for _ in range(num_rooms)]
	ac_prev = 1000

	damper_cycles = [0 for _ in range(num_rooms)]
	ac_cycle = 0

	change_temp = set([0] + [random.randrange(0, sim_max) for _ in range(num_setpoints - 1)])

	rooms: List[housebuilder.Room] = house.get_rooms(0)

	for t in range(sim_max):
		if t in change_temp:
			for room in rooms:
				room.set_setpoint(random.uniform(20, 28))
				
		for i in range(num_rooms):
			temps[i][t] = rooms[i].get_temp()
			setps[i][t] = rooms[i].get_setpoint()

		outside_temp[t] = const.OUTSIDE_TEMP[weather_start + t]

		ac_status, dampers_a = agent(house, const.OUTSIDE_TEMP[weather_start + t])
		house.step(const.OUTSIDE_TEMP[weather_start + t], ac_status, dampers_a)
		
		for i in range(num_rooms):
			total_devs[i] += abs(rooms[i].get_temp() - rooms[i].get_setpoint())
			devs[i][t] = total_devs[i] / (t + 1)
			dampers[i][t] = 1 if dampers_a[0][i] else 0
			if dampers[i][t] != damper_prevs[i]:
				damper_cycles[i] += 1
			damper_prevs[i] = dampers[i][t]

		ac_power[t] = ac_status
		if ac_power[t] != ac_prev:
			ac_cycle += 1
		ac_prev = ac_power[t]

	for i in range(num_rooms):
		ax_left[i].plot(xvalues, temps[i], color="red", linewidth=0.1)
		ax_left[i].plot(xvalues, setps[i], color="blue", linewidth=0.5)
		ax_right[i].plot(xvalues, devs[i], color="purple", linewidth=0.5)
		ax_right[i].plot(xvalues, dampers[i], color="gray", linewidth=0.2)
		ax_left[i].plot(xvalues, outside_temp, color="green", linewidth=0.1)
		print(f"room {i}: damper cycles {damper_cycles[i]} dev {total_devs[i] / sim_max}")
	print("ac cycles", ac_cycle)

	status_ax.plot(xvalues, ac_power, linewidth=0.1)

	fig.set_size_inches(9.6, 4.8 / 2 * (num_rooms * 2 + 1))
	plt.savefig(args.output, dpi=500, bbox_inches="tight")
	
if __name__ == "__main__":
	main()