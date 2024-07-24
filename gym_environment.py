import numpy as np
import random
import gymnasium as gym
import housebuilder
import torch
import const
import math

class Environment(gym.Env):
	def __init__(self, house_cfg: str):
		self.observation_space = gym.spaces.Box(-10, 70, shape=(5,), dtype=float)
		self.action_space = gym.spaces.Discrete(3)
		self._house_cfg = house_cfg

	def _get_observations(self):
		return torch.tensor([
			self.house.get_rooms(0)[0].get_temp(),
			self.house.get_rooms(0)[0].get_setpoint(),
			self.house.get_rooms(0)[1].get_temp(),
			self.house.get_rooms(0)[1].get_setpoint(),
			const.OUTSIDE_TEMP[self._weather_start + self._time]
		])
	
	def _get_reward(self):
		# L2 norm
		return -math.sqrt(
			(self.house.get_rooms(0)[0].get_temp() - self.house.get_rooms(0)[0].get_setpoint()) ** 2 + 
			(self.house.get_rooms(0)[1].get_temp() - self.house.get_rooms(0)[1].get_setpoint()) ** 2
		)
	
	def reset(self, num_setpoints=1, length=1440, weather_start=None):
		super().reset()

		self.house = housebuilder.build_house(self._house_cfg)
		self._time = 0
		self._length = length
		self._weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - length) if weather_start is None else weather_start
		self._change_setpoint = set([random.randrange(0, length) for _ in range(num_setpoints - 1)])

		self.house.get_rooms(0)[0].set_setpoint(random.uniform(20, 28))
		self.house.get_rooms(0)[1].set_setpoint(random.uniform(20, 28))

		return self._get_observations(), self._get_reward()

	def step(self, power: int, dampers: list[list[bool]]):
		self.house.step(const.OUTSIDE_TEMP[self._weather_start + self._time], power, dampers)
		reward = self._get_reward()
		terminated = self._time > self._length
		self._time += 1		
		return self._get_observations(), reward, terminated
