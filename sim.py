import const
import agents.dumb_agent
import agents.pid_agent
import random
import numpy as np
import matplotlib.pyplot as plt

room_air_mass = const.ROOM_LENGTH * const.ROOM_WIDTH * const.ROOM_HEIGHT * const.AIR_DENSITY
wall_area_sum = 2 * (const.ROOM_LENGTH * const.ROOM_HEIGHT + const.ROOM_WIDTH * const.ROOM_HEIGHT)
roof_area = (const.ROOM_LENGTH * const.ROOM_WIDTH)
cool_energy_transfer_watt = const.COOL_BTUS / 3.41
heat_energy_transfer_watt = const.HEAT_BTUS / 3.41

# power_transfers = [cool_energy_transfer_watt, heat_energy_transfer_watt]

def clamp(val: float, min: float, max: float) -> float:
	if val < min:
		return min
	if val > max:
		return max
	return val

def joule_to_temp_change(joule: float) -> float:
	return joule / (room_air_mass * const.AIR_HEAT_CAPACITY)

# convection heat transfer equation Q = hA(Delta)T

# calculate convection from outside air to outer wall
def calc_convection_to_ext_wall(out_wall_temp: float, time: float) -> float:
	change = const.OUTSIDE_CONVECTION_COEFF * wall_area_sum * (const.OUTSIDE_TEMP[time] - out_wall_temp) / 2
	return joule_to_temp_change(change * 60)

# calculate conduction transfer from outside to inside of wall
def calc_wall_conduction(int_wall_temp: float, out_wall_temp: float) -> float:
	# TODO possible investigate switching int_wall_temp and out_wall_temp
	change = wall_area_sum * (out_wall_temp - int_wall_temp) * const.WALL_THERM_COND / const.WALL_THICK
	return joule_to_temp_change(change * 60)

# calculate convection from inner wall to room air
def calc_convection_to_room(room_temp: float, int_wall_temp: float) -> float:
	change = const.INSIDE_CONVECTION_COEFF * wall_area_sum * (int_wall_temp - room_temp)
	return joule_to_temp_change(change * 60)

# def calc_transfer_into_roof(in_temp: float, out_temp: float) -> float:
# 	change = const.OUTSIDE_CONVECTION_COEFF * roof_area * (out_temp - in_temp)/2
# 	return joule_to_temp_change(change * 60)

# def calc_next_roof_temp(cur_roof_temp: float, time: float) -> float:
# 	change = calc_transfer_into_roof(cur_roof_temp, const.OUTSIDE_TEMP[time])
# 	return cur_roof_temp + change

# def calc_transfer_thru_roof(in_temp: float, cur_roof_temp: float) -> float:
# 	change = roof_area * (in_temp - cur_roof_temp) * const.ROOF_THERM_COND / const.ROOF_THICK
# 	return joule_to_temp_change(change * 60)

# def calc_next_int_roof_temp(cur_temp: float, cur_roof_temp: float) -> float:
# 	change = calc_transfer_thru_roof(cur_temp, cur_roof_temp)
# 	return cur_roof_temp + change

# def calc_transfer_roof_to_air(in_temp: float, cur_int_roof_temp: float) -> float:
# 	change = const.INSIDE_CONVECTION_COEFF * roof_area * (cur_int_roof_temp - in_temp)
# 	return joule_to_temp_change(change * 60)


# def calc_transfer_thru_roof(in_temp: float, out_temp: float) -> float:
# 	change = roof_area * (out_temp - in_temp) * const.ROOF_THERM_COND / const.ROOF_THICK
# 	return joule_to_temp_change(change * 60)

# -1 <= power <= 1
# -1 = full cool, 1 = full heat
def calc_ac_effect(power: float) -> float:
	if power == 0:
		return 0
	change = cool_energy_transfer_watt if power < 0 else heat_energy_transfer_watt
	random_tensor = random.uniform(const.NOISE_MULT_MIN, const.NOISE_MULT_MAX)
	return joule_to_temp_change(change * random_tensor * 60)

def calc_next_temp(power: float, room_temp: float, int_wall_temp: float) -> float:
	change1 = calc_convection_to_room(room_temp, int_wall_temp)
	change2 = calc_ac_effect(power)
	return room_temp + change1 + change2


# def calc_next_temp(power: float, cur_temp: float, time: float) -> float:
# 	change = calc_transfer_thru_wall(cur_temp, torch.take(const.OUTSIDE_TEMP, time))
# 	change += calc_transfer_thru_roof(cur_temp, torch.take(const.OUTSIDE_TEMP, time))
# 	# power = clamp(power, -1, 1)

# 	# if power < 0 and not const.COOL_IS_CONTINUOUS:
# 	# 	power = torch.round(power * (const.COOL_SETTINGS_NUM - 1)) / (const.COOL_SETTINGS_NUM - 1)
# 	# if power > 0 and not const.HEAT_IS_CONTINUOUS:
# 	# 	power = torch.round(power * (const.HEAT_SETTINGS_NUM - 1)) / (const.HEAT_SETTINGS_NUM - 1)
	
# 	# if power < 0 and power > -const.COOL_MIN_POWER:
# 	# 	power = 0
# 	# if power > 0 and power < const.HEAT_MIN_POWER:
# 	# 	power = 0

# 	change += calc_ac_effect(power)
# 	return cur_temp + change

# if __name__ == "__main__":
# 	random.seed(1)

# 	fig, ax1 = plt.subplots()
# 	ax1.set_xlabel("time (min)")
# 	ax1.set_ylim(10, 30)
# 	ax1.set_yticks(np.arange(10, 31))
# 	ax1.set_ylabel("deg C")

# 	ax2 = ax1.twinx()
# 	ax2.set_ylim(-1, 5)
# 	ax2.set_ylabel("mean temp deviation or ac/heater power")

# 	target_temperature = 0

# 	cur_temp = const.ROOM_START_TEMP

# 	sim_max = 1440

# 	xvalues = np.arange(0, sim_max)
# 	temperatures = np.zeros(sim_max)
# 	setpoints = np.zeros(sim_max)
# 	outside_temp = np.zeros(sim_max)
# 	on_off = np.zeros(sim_max)
# 	mean_deviation = np.zeros(sim_max)

# 	deviation_sum = 0
# 	old_power = 0
# 	cycle_count = 0

# 	for i in range(sim_max):
# 		if i in const.SETPOINT_LIST:
# 			target_temperature = const.SETPOINT_LIST[i]
# 		temperatures[i] = cur_temp
# 		setpoints[i] = target_temperature
# 		change = calc_transfer_thru_wall(cur_temp, const.OUTSIDE_TEMP[i])
# 		power = agents.dumb_agent.agent(cur_temp, const.OUTSIDE_TEMP[i], target_temperature, old_power)
# 		power = clamp(power, -1, 1)

# 		if power < 0 and not const.COOL_IS_CONTINUOUS:
# 			power = round(power * (const.COOL_SETTINGS_NUM - 1)) / (const.COOL_SETTINGS_NUM - 1)
# 		if power > 0 and not const.HEAT_IS_CONTINUOUS:
# 			power = round(power * (const.HEAT_SETTINGS_NUM - 1)) / (const.HEAT_SETTINGS_NUM - 1)
		
# 		if power < 0 and power > -const.COOL_MIN_POWER:
# 			power = 0
# 		if power > 0 and power < const.HEAT_MIN_POWER:
# 			power = 0

# 		if (old_power < 0 or old_power > 0) and power == 0:
# 			cycle_count += 1

# 		old_power = power
# 		change += calc_ac_effect(power)
# 		cur_temp += change
# 		outside_temp[i] = const.OUTSIDE_TEMP[i]
# 		on_off[i] = power

# 		deviation_sum += abs(cur_temp - target_temperature)
# 		mean_deviation[i] = deviation_sum / (i + 1)
# 		# print(f"current {cur_temp} want {target_temperature}")

# 	print(f"cycle count {cycle_count}")

# 	ax1.plot(xvalues, temperatures, color="red", linewidth=0.1)
# 	ax1.plot(xvalues, setpoints, color="blue", linewidth=0.1)
# 	ax1.plot(xvalues, outside_temp, color="green", linewidth=0.1)
# 	ax2.plot(xvalues, on_off, color="black", linewidth=0.05)
# 	ax2.plot(xvalues, mean_deviation, color="purple", linewidth=0.1)
# 	plt.savefig("stupid.png", dpi=1000)

# 	# m2K/W * m2 * K
