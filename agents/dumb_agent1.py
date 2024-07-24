import housebuilder

constants = housebuilder.get_constants()

def prelim_agent(house: housebuilder.House, room0_temp: float, room0_setp: float):
	if room0_temp > room0_setp + constants.epsilon:
		return (constants.settings.index(-1), [[False, False]])
	if room0_temp < room0_setp - constants.epsilon:
		return (constants.settings.index(1), [[False, False]])
	return (house.ac_status, house.dampers)

def agent(house: housebuilder. House, outside_temp: float):
	room0_temp = house.rooms[0][0].get_temp()
	room0_setp = house.rooms[0][0].get_setpoint()

	ac_status, dampers = prelim_agent(house, room0_temp, room0_setp)
	if room0_temp > outside_temp and constants.settings[ac_status] < 0:
		ac_status = constants.settings.index(0)
		dampers = house.dampers
	if room0_temp < outside_temp and constants.settings[ac_status] > 0:
		ac_status = constants.settings.index(0)
		dampers = house.dampers
	
	return (ac_status, dampers)
