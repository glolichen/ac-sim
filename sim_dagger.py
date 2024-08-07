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
import gym_environment
import imitation
import imitation.policies
import imitation.policies.base
import enum
from typing import Union, Dict

parser = argparse.ArgumentParser(prog="SimulatorDAgger", add_help=False)
parser.add_argument("-o", "--output")
parser.add_argument("-t", "--time")
parser.add_argument("-m", "--model")

def main():
	global num
	args = parser.parse_args()
	if args.output == None:
		args.output = "out/out.png"
		print("warn: no output passed, default to out/out.png")
	if args.time == None:
		args.time = 1440
		print("warn: no time passed, default to 1440")
	if args.model == None:
		args.model = "models/dagger_out.zip"
		print("warn: no time passed, default to models/dagger_out.zip")

	env = gym_environment.Environment()

	num_rooms = 5

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

	class DumbPolicy(imitation.policies.base.NonTrainablePolicy):
		class Status(enum.Enum):
			NEED_COOL = 0
			WANT_COOL = 1
			NEED_HEAT = 2
			WANT_HEAT = 3
			EQUAL = 4
		def _choose_action(self, obs: Union[np.ndarray, Dict[str, np.ndarray]],) -> int:
			num_rooms = int((len(obs) - 3) / 4)

			epsilon = 0.9
			statuses = []
			need_heat, need_cool = 0, 0
			badness_heat, badness_cool = 0, 0
			min_temp, max_temp = 100000, -100000

			old_ac_status = obs[num_rooms * 2 + 2]
			old_dampers = [[]]
			for i in range(num_rooms):
				old_dampers[0].append(obs[num_rooms * 2 + 4 + i * 2])

			for i in range(num_rooms):
				temp, setp = obs[i * 2], obs[i * 2 + 1]
				min_temp = min(min_temp, temp)
				max_temp = max(max_temp, temp)
				if temp < setp - epsilon:
					statuses.append(self.Status.NEED_HEAT.value)
					badness_heat += abs(temp - setp)
					need_heat += 1
				elif temp < setp:
					statuses.append(self.Status.WANT_HEAT.value)
					badness_heat += abs(temp - setp)
				elif temp > setp + epsilon:
					statuses.append(self.Status.NEED_COOL.value)
					badness_cool += abs(temp - setp)
					need_cool += 1
				elif temp > setp:
					statuses.append(self.Status.WANT_COOL.value)
					badness_cool += abs(temp - setp)
				else:
					statuses.append(self.Status.EQUAL.value)

			outside_temp = obs[num_rooms * 2]
			dampers = [[]]
			if need_heat > need_cool:
				if max_temp >= outside_temp:
					for status in statuses:
						if status == self.Status.NEED_HEAT.value or status == self.Status.WANT_HEAT.value:
							dampers[0].append(False)
						else:
							dampers[0].append(True)
					return env.actions.index((1, dampers))
				else:
					return env.actions.index((0, old_dampers))
			if need_cool > need_heat:
				if min_temp <= outside_temp:
					for status in statuses:
						if status == self.Status.NEED_COOL.value or status == self.Status.WANT_COOL.value:
							dampers[0].append(False)
						else:
							dampers[0].append(True)
					return env.actions.index((-1, dampers))
				else:
					return env.actions.index((0, old_dampers))

			if need_cool > 0 and need_heat > 0:
				if badness_cool > badness_heat:
					if min_temp <= outside_temp:
						for status in statuses:
							if status == self.Status.NEED_COOL.value or status == self.Status.WANT_COOL.value:
								dampers[0].append(False)
							else:
								dampers[0].append(True)
						return env.actions.index((-1, dampers))
					else:
						return env.actions.index((0, old_dampers))
				if badness_heat > badness_cool:
					if max_temp >= outside_temp:
						for status in statuses:
							if status == self.Status.NEED_HEAT.value or status == self.Status.WANT_HEAT.value:
								dampers[0].append(False)
							else:
								dampers[0].append(True)
						return env.actions.index((1, dampers))
					else:
						return env.actions.index((0, old_dampers))
				
			return env.actions.index((old_ac_status, old_dampers))

	
	# model = DumbPolicy(env.observation_space, env.action_space)
	model = imitation.policies.base.FeedForward32Policy.load(args.model)

	obs, _ = env.reset(num_setpoints=num_setpoints, length=sim_max, weather_start=weather_start)

	rooms: List[housebuilder.Room] = env.house.get_rooms(0)

	for t in range(sim_max):
		action, _ = model.predict(obs, deterministic=True)
		ac_status, dampers_a = env.get_action(action)
		obs, _, terminated, _, _ = env.step(action)
		
		for i in range(num_rooms):
			temps[i][t] = rooms[i].get_temp()
			setps[i][t] = rooms[i].get_setpoint()

		outside_temp[t] = const.OUTSIDE_TEMP[weather_start + t]

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