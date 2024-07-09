Kp = 0.3
Ki = 0.005
# # Ki = 0
# Ki = 0
Kd = 0.009

old_error = 0
integral = 0

def agent(in_temp: float, out_temp: float, target_temp: float) -> float:
	global old_error
	global integral
	error = target_temp - in_temp
	if -0.1 <= error <= 0.1:
		return 0
	output = error * Kp # prop
	integral += error
	output += integral * Ki # integral
	output += (error - old_error) * Kd # derivative
	old_error = error
	return output