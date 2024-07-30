import housebuilder
def agent(house: housebuilder. House, outside_temp: float):
	room0_temp = house.rooms[0][0].get_temp()
	room0_setp = house.rooms[0][0].get_setpoint()
	room1_temp = house.rooms[0][1].get_temp()
	room1_setp = house.rooms[0][1].get_setpoint()

	if room0_temp > room0_setp + house.constants.epsilon and room1_temp > room1_setp:
		ac_status, dampers = (house.constants.settings.index(-1), [[False, False]])
	elif room0_temp < room0_setp - house.constants.epsilon and room1_temp < room1_setp:
		ac_status, dampers = (house.constants.settings.index(1), [[False, False]])
	
	elif room0_temp > room0_setp and room1_temp > room1_setp + house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(-1), [[False, False]])
	elif room0_temp < room0_setp and room1_temp < room1_setp - house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(1), [[False, False]])

	elif room0_temp > room0_setp + house.constants.epsilon and abs(room1_temp - room1_setp) < house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(-1), [[False, True]])
	elif room1_temp > room1_setp + house.constants.epsilon and abs(room0_temp - room0_setp) < house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(-1), [[True, False]])
	
	elif room0_temp < room0_setp - house.constants.epsilon and abs(room1_temp - room1_setp) < house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(1), [[False, True]])
	elif room1_temp < room1_setp - house.constants.epsilon and abs(room0_temp - room0_setp) < house.constants.epsilon:
		ac_status, dampers = (house.constants.settings.index(1), [[True, False]])
	
	elif room0_temp > room0_setp + house.constants.epsilon and room1_temp < room1_setp - house.constants.epsilon:
		error0, error1 = abs(room0_temp - room0_setp), abs(room1_temp - room1_setp)
		if error0 > error1:
			ac_status, dampers = (house.constants.settings.index(-1), [[False, True]])
		else:
			ac_status, dampers = (house.constants.settings.index(1), [[True, False]])
	elif room0_temp < room0_setp - house.constants.epsilon and room1_temp > room1_setp + house.constants.epsilon:
		error0, error1 = abs(room0_temp - room0_setp), abs(room1_temp - room1_setp)
		if error0 > error1:
			ac_status, dampers = (house.constants.settings.index(1), [[False, True]])
		else:
			ac_status, dampers = (house.constants.settings.index(-1), [[True, False]])
	# if abs(room0_temp - room0_setp) < constants.epsilon and abs(room1_temp - room1_setp) < constants.epsilon:
	# 	return (constants.settings.index(0), house.dampers)
	else:
		ac_status, dampers = (house.ac_status, house.dampers)
	
	if room0_temp > outside_temp and room1_temp > outside_temp and house.constants.settings[ac_status] < 0:
		ac_status = house.constants.settings.index(0)
		dampers = house.dampers
	if room0_temp < outside_temp and room1_temp < outside_temp and house.constants.settings[ac_status] > 0:
		ac_status = house.constants.settings.index(0)
		dampers = house.dampers
	
	return (ac_status, dampers)
