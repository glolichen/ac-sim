import torch

OUTSIDE_TEMP = []
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
