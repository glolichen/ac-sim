import housebuilder
import enum

class Status(enum.Enum):
	NEED_COOL = 0
	WANT_COOL = 1
	NEED_HEAT = 2
	WANT_HEAT = 3
	EQUAL = 4

def agent(house: housebuilder. House, outside_temp: float):
	epsilon: float = house.constants.epsilon
	rooms: list[housebuilder.Room] = house.get_rooms(0)
	statuses = []
	need_heat, need_cool = 0, 0
	badness_heat, badness_cool = 0, 0
	min_temp, max_temp = 100000, -100000
	for room in rooms:
		temp, setp = room.get_temp(), room.get_setpoint()
		min_temp = min(min_temp, temp)
		max_temp = max(max_temp, temp)
		if temp < setp - epsilon:
			statuses.append(Status.NEED_HEAT.value)
			badness_heat += abs(temp - setp)
			need_heat += 1
		elif temp < setp:
			statuses.append(Status.WANT_HEAT.value)
			badness_heat += abs(temp - setp)
		elif temp > setp + epsilon:
			statuses.append(Status.NEED_COOL.value)
			badness_cool += abs(temp - setp)
			need_cool += 1
		elif temp > setp:
			statuses.append(Status.WANT_COOL.value)
			badness_cool += abs(temp - setp)
		else:
			statuses.append(Status.EQUAL.value)

	
	dampers = [[]]
	if need_heat > need_cool:
		if max_temp >= outside_temp:
			for status in statuses:
				if status == Status.NEED_HEAT.value or status == Status.WANT_HEAT.value:
					dampers[0].append(False)
				else:
					dampers[0].append(True)
			return (1, dampers)
		else:
			return (0, house.dampers)
	if need_cool > need_heat:
		if min_temp <= outside_temp:
			for status in statuses:
				if status == Status.NEED_COOL.value or status == Status.WANT_COOL.value:
					dampers[0].append(False)
				else:
					dampers[0].append(True)
			return (-1, dampers)
		else:
			return (0, house.dampers)

	if need_cool > 0 and need_heat > 0:
		if badness_cool > badness_heat:
			if min_temp <= outside_temp:
				for status in statuses:
					if status == Status.NEED_COOL.value or status == Status.WANT_COOL.value:
						dampers[0].append(False)
					else:
						dampers[0].append(True)
				return (-1, dampers)
			else:
				return (0, house.dampers)
		if badness_heat > badness_cool:
			if max_temp >= outside_temp:
				for status in statuses:
					if status == Status.NEED_HEAT.value or status == Status.WANT_HEAT.value:
						dampers[0].append(False)
					else:
						dampers[0].append(True)
				return (1, dampers)
			else:
				return (0, house.dampers)
		
	# todo
	return (house.ac_status, house.dampers)