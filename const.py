import torch

# meters
ROOM_LENGTH = 4
ROOM_WIDTH = 5
ROOM_HEIGHT = 3

COOL_BTUS = 8600 # AC unit BTUs per second
HEAT_BTUS = 9600 # AC unit BTUs per second

AIR_DENSITY = 1.204 # kg/m3
AIR_HEAT_CAPACITY = 718 # J/kgC

EXT_WALL_THICK = 0.20 # thickness (m)
EXT_WALL_THERM_COND = 1000000000#  0.04 # thermal conductivity (W/mK)

INT_WALL_THICK = 0.12
INT_WALL_THERM_COND = 1000000000 # 0.18

OUT_WALL_STARTING_TEMP = 20

OUTSIDE_CONVECTION_COEFF = 13 # W/m^2 * C -----  affected by circulation of environment, may have to change later
INSIDE_CONVECTION_COEFF = 2 # W/m^2 * C -----  affected by circulation of environment, may have to change later

ROOF_THICK = 0.12 # thickness (m)
ROOF_THERM_COND = 0.41 # thermal conductivity (W/mK)

FLOOR_AREA = 20 # meters same as roof


ROOM_START_TEMP = 20
OUTSIDE_TEMP = []
COMFORT_TOLERANCE = 1

DETERMINISTIC = True
NOISE_MULT_MIN = 0.8
NOISE_MULT_MAX = 1.2

COOL_IS_CONTINUOUS = True
HEAT_IS_CONTINUOUS = False

# if not continuous, use these values
COOL_SETTINGS_NUM = 5 # 0, 0.25, 0.5, 0.75, 1
HEAT_SETTINGS_NUM = 2 # 0, 1

COOL_MIN_POWER = 0.25
HEAT_MIN_POWER = 0.25

WEATHER_FILES = [
	"weather1.csv",
	"weather2.csv",
]

DEVICE = torch.device("cpu")

for file in WEATHER_FILES:
	with open(file, "r") as file:
		for line in file.readlines():
			temp = float(line.split(",")[3])
			temp = (temp - 32) / 9 * 5
			OUTSIDE_TEMP.append(temp)

DAMPER_LEAK = 0.05
REWARD_CONSTANT = 0.1353352832