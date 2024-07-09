import const 

def agent(in_temp: float, out_temp: float, target_temp: float, last_power: float) -> float:
	# cool
	if last_power < 0 and in_temp > target_temp - const.COMFORT_TOLERANCE and out_temp > in_temp:
		return last_power
	# heat
	if last_power > 0 and in_temp < target_temp + const.COMFORT_TOLERANCE and out_temp < in_temp:
		return last_power


	if in_temp > target_temp + const.COMFORT_TOLERANCE and out_temp > in_temp:
		return -1
	if in_temp < target_temp - const.COMFORT_TOLERANCE and out_temp < in_temp:
		return 1
	return 0