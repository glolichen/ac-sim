import const
import agents.dumb_agent
import math
import agents.pid_agent
import random
import numpy as np
import matplotlib.pyplot as plt

room_air_mass = const.ROOM_LENGTH * const.ROOM_WIDTH * const.ROOM_HEIGHT * const.AIR_DENSITY
wall_area_sum = 2 * (const.ROOM_LENGTH * const.ROOM_HEIGHT + const.ROOM_WIDTH * const.ROOM_HEIGHT)
roof_area = (const.ROOM_LENGTH * const.ROOM_WIDTH)
cool_energy_transfer_watt = const.COOL_BTUS / 3.41
heat_energy_transfer_watt = const.HEAT_BTUS / 3.41

def sgn(val: float) -> int:
	if val < 0:
		return -1
	if val > 0:
		return 1
	return 0
def clamp(val: float, min: float, max: float) -> float:
	if val < min:
		return min
	if val > max:
		return max
	return val

def joule_to_temp_change(joule: float) -> float:
	return joule / (room_air_mass * const.AIR_HEAT_CAPACITY)

def calc_transfer_thru_wall(in_temp: float, out_temp: float) -> float:
	change = (wall_area_sum * (out_temp - in_temp) * const.WALL_THERM_COND / const.WALL_THICK) + (const.WALL_EMISSIVITY * const.STEFAN_BOLTZMANN_CONSTANT * wall_area_sum)
	return joule_to_temp_change(change * 60)

def calc_transfer_thru_roof(in_temp: float, out_temp: float) -> float:
	change = (roof_area * (out_temp - in_temp) * const.ROOF_THERM_COND / const.ROOF_THICK) + (const.ROOF_EMISSIVITY * const.STEFAN_BOLTZMANN_CONSTANT * roof_area)
	return joule_to_temp_change(change * 60)

# -1 <= power <= 1
# -1 = full cool, 1 = full heat
def calc_ac_effect(power: float) -> float:	
	# print(f"{power} ({'HEAT' if power > 0 else ('COOL' if power < 0 else 'OFF')})")
	change = (cool_energy_transfer_watt if power < 0 else heat_energy_transfer_watt) * power
	return joule_to_temp_change(change * 60 * random.uniform(const.NOISE_MULT_MIN, const.NOISE_MULT_MAX))

def calc_next_temp(power: float, cur_temp: float, time: int) -> float:
	change = calc_transfer_thru_wall(cur_temp, const.OUTSIDE_TEMP[time])
	change += calc_transfer_thru_roof(cur_temp, const.OUTSIDE_TEMP[time])
	power = clamp(power, -1, 1)

	if power < 0 and not const.COOL_IS_CONTINUOUS:
		power = round(power * (const.COOL_SETTINGS_NUM - 1)) / (const.COOL_SETTINGS_NUM - 1)
	if power > 0 and not const.HEAT_IS_CONTINUOUS:
		power = round(power * (const.HEAT_SETTINGS_NUM - 1)) / (const.HEAT_SETTINGS_NUM - 1)
	
	if power < 0 and power > -const.COOL_MIN_POWER:
		power = 0
	if power > 0 and power < const.HEAT_MIN_POWER:
		power = 0

	change += calc_ac_effect(power)
	return cur_temp + change

if __name__ == "__main__":
	random.seed(1)

	fig, ax1 = plt.subplots()
	ax1.set_xlabel("time (min)")
	ax1.set_ylim(10, 30)
	ax1.set_yticks(np.arange(10, 31))
	ax1.set_ylabel("deg C")

	ax2 = ax1.twinx()
	ax2.set_ylim(-1, 5)
	ax2.set_ylabel("mean temp deviation or ac/heater power")

	target_temperature = 0

	cur_temp = const.ROOM_START_TEMP

	sim_max = 1440

	xvalues = np.arange(0, sim_max)
	temperatures = np.zeros(sim_max)
	setpoints = np.zeros(sim_max)
	outside_temp = np.zeros(sim_max)
	on_off = np.zeros(sim_max)
	mean_deviation = np.zeros(sim_max)

	deviation_sum = 0
	old_power = 0
	cycle_count = 0

	for i in range(sim_max):
		if i in const.SETPOINT_LIST:
			target_temperature = const.SETPOINT_LIST[i]
		temperatures[i] = cur_temp
		setpoints[i] = target_temperature
		change = calc_transfer_thru_wall(cur_temp, const.OUTSIDE_TEMP[i])
		power = agents.dumb_agent.agent(cur_temp, const.OUTSIDE_TEMP[i], target_temperature, old_power)
		power = clamp(power, -1, 1)

		if power < 0 and not const.COOL_IS_CONTINUOUS:
			power = round(power * (const.COOL_SETTINGS_NUM - 1)) / (const.COOL_SETTINGS_NUM - 1)
		if power > 0 and not const.HEAT_IS_CONTINUOUS:
			power = round(power * (const.HEAT_SETTINGS_NUM - 1)) / (const.HEAT_SETTINGS_NUM - 1)
		
		if power < 0 and power > -const.COOL_MIN_POWER:
			power = 0
		if power > 0 and power < const.HEAT_MIN_POWER:
			power = 0

		if (old_power < 0 or old_power > 0) and power == 0:
			cycle_count += 1

		old_power = power
		change += calc_ac_effect(power)
		cur_temp += change
		outside_temp[i] = const.OUTSIDE_TEMP[i]
		on_off[i] = power

		deviation_sum += abs(cur_temp - target_temperature)
		mean_deviation[i] = deviation_sum / (i + 1)
		# print(f"current {cur_temp} want {target_temperature}")

	print(f"cycle count {cycle_count}")

	ax1.plot(xvalues, temperatures, color="red", linewidth=0.1)
	ax1.plot(xvalues, setpoints, color="blue", linewidth=0.1)
	ax1.plot(xvalues, outside_temp, color="green", linewidth=0.1)
	ax2.plot(xvalues, on_off, color="black", linewidth=0.05)
	ax2.plot(xvalues, mean_deviation, color="purple", linewidth=0.1)
	plt.savefig("stupid.png", dpi=1000)

	# m2K/W * m2 * K