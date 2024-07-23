import housebuilder

KP = 0.3
KI = 0.001
KD = 0
FF = 0

constants = housebuilder.get_constants()
def get_closest_setting(power: float):
	best, best_error = -1, 10000
	for i in range(len(constants.settings)):
		error = abs(constants.settings[i] - power)
		if error < best_error:
			best, best_error = i, error
	return best

class PIDController:
	def __init__(self, KP: float, KI: float, KD: float, FF: float):
		self.KP, self.KI, self.KD, self.FF = KP, KI, KD, FF
		self.integral = 0
		self.old_error = None
	def calc(self, error: float):
		self.integral += error
		P = error * self.KP
		I = self.integral * self.KI
		D = (error - self.old_error) * self.KD if self.old_error is not None else 0
		self.old_error = error
		return get_closest_setting(P + I + D + self.FF)
	
pid = PIDController(KP, KI, KD, FF)

def prelim_agent(house: housebuilder.House, room0_temp: float, room0_setp: float, room1_temp: float, room1_setp: float):
	error0 = room0_setp - room0_temp
	error1 = room1_setp - room1_temp
	avg_error = (error0 + error1) / 2

	if room0_temp > room0_setp + constants.epsilon and room1_temp > room1_setp + constants.epsilon:
		return (pid.calc(avg_error), [[False, False]])
	if room0_temp < room0_setp - constants.epsilon and room1_temp < room1_setp - constants.epsilon:
		return (pid.calc(avg_error), [[False, False]])

	if room0_temp > room0_setp + constants.epsilon and abs(error1) < constants.epsilon:
		return (pid.calc(error0), [[False, True]])
	if room1_temp > room1_setp + constants.epsilon and abs(error0) < constants.epsilon:
		return (pid.calc(error1), [[True, False]])
	
	if room0_temp < room0_setp - constants.epsilon and abs(error1) < constants.epsilon:
		return (pid.calc(error0), [[False, True]])
	if room1_temp < room1_setp - constants.epsilon and abs(error0) < constants.epsilon:
		return (pid.calc(error1), [[True, False]])
	
	if room0_temp > room0_setp + constants.epsilon and room1_temp < room1_setp - constants.epsilon:
		if abs(error0) > abs(error1):
			return (pid.calc(error0), [[False, True]])
		else:
			return (pid.calc(error1), [[True, False]])
	if room0_temp < room0_setp - constants.epsilon and room1_temp > room1_setp + constants.epsilon:
		if abs(error0) > abs(error1):
			return (pid.calc(error0), [[False, True]])
		else:
			return (pid.calc(error1), [[True, False]])
	
	if abs(error0) < constants.epsilon and abs(error1) < constants.epsilon:
		return (constants.settings.index(0), house.dampers)

	return (house.ac_status, house.dampers)

def agent(house: housebuilder. House, outside_temp: float):
	room0_temp = house.rooms[0][0].get_temp()
	room0_setp = house.rooms[0][0].get_setpoint()
	room1_temp = house.rooms[0][1].get_temp()
	room1_setp = house.rooms[0][1].get_setpoint()

	ac_status, dampers = prelim_agent(house, room0_temp, room0_setp, room1_temp, room1_setp)
	if room0_temp > outside_temp and room1_temp > outside_temp and constants.settings[ac_status] < 0:
		ac_status = constants.settings.index(0)
		dampers = house.dampers
	if room0_temp < outside_temp and room1_temp < outside_temp and constants.settings[ac_status] > 0:
		ac_status = constants.settings.index(0)
		dampers = house.dampers
	
	return (ac_status, dampers)