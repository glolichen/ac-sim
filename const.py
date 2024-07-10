import torch

# meters
ROOM_LENGTH = 4
ROOM_WIDTH = 5
ROOM_HEIGHT = 3

COOL_BTUS = 6000 # AC unit BTUs per second
HEAT_BTUS = 4000 # AC unit BTUs per second

AIR_DENSITY = 1.204 # kg/m3
AIR_HEAT_CAPACITY = 718 # J/kgC

WALL_EMISSIVITY = 0.94 # no clue
WALL_THICK = 0.12 # thickness (m)
WALL_THERM_COND = 0.04 # thermal conductivity (W/mK)

STEFAN_BOLTZMANN_CONSTANT = 0.000000567 # part of the heat transfer equation

ROOF_EMISSIVITY = 0.8 # no clue
ROOF_THICK = 0.05 # thickness (m)
ROOF_THERM_COND = 0.062 # thermal conductivity (W/mK)

ROOM_START_TEMP = 20
OUTSIDE_TEMP = []
COMFORT_TOLERANCE = 0.45

# NOISE_MULT_MIN = 0.8
# NOISE_MULT_MAX = 1.2
NOISE_MULT_MIN = 1
NOISE_MULT_MAX = 1

COOL_IS_CONTINUOUS = True
HEAT_IS_CONTINUOUS = False

# if not continuous, use these values
COOL_SETTINGS_NUM = 5 # 0, 0.25, 0.5, 0.75, 1
HEAT_SETTINGS_NUM = 2 # 0, 1

COOL_MIN_POWER = 0.25
HEAT_MIN_POWER = 0.25

SETPOINT_LIST = {
	0: 23,
	200: 20,
	900: 28,
	1100: 21
}

WEATHER_FILES = [
	"weather1.csv",
	"weather2.csv",
]

for file in WEATHER_FILES:
	with open(file, "r") as file:
		for line in file.readlines():
			temp = float(line.split(",")[3])
			temp = (temp - 32) / 9 * 5
			OUTSIDE_TEMP.append(temp)

OUTSIDE_TEMP = torch.tensor(OUTSIDE_TEMP, device="cuda:0")