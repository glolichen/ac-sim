import numpy as np
import random
import gymnasium as gym
import housebuilder
import const

class Environment(gym.Env):
	def __init__(self):
		self.observation_space = gym.spaces.Box(-10, 400, shape=(11,), dtype=float)
		self.actions = [
			(-1, [[False, False]]),
			(-1, [[False, True]]),
			(-1, [[True, False]]),
			(-1, [[True, True]]),
			(0 , [[False, False]]),
			(0 , [[False, True]]),
			(0 , [[True, False]]),
			(0 , [[True, True]]),
			(1 , [[False, False]]),
			(1 , [[False, True]]),
			(1 , [[True, False]]),
			(1 , [[True, True]]),
		]
		self.action_space = gym.spaces.Discrete(len(self.actions))
		self._house_cfg = "2r_simple.json"

	def _get_observations(self):
		return np.array([
			self.house.get_rooms(0)[0].get_temp(),
			self.house.get_rooms(0)[0].get_setpoint(),
			self.house.get_rooms(0)[1].get_temp(),
			self.house.get_rooms(0)[1].get_setpoint(),
			const.OUTSIDE_TEMP[self._weather_start + self._time],
			self._ac_last_change,
			self._prev_ac,
			self._damper0_last_change,
			self._prev_damper0,
			self._damper1_last_change,
			self._prev_damper1
		])
	
	def _get_reward(self):
		# L2 norm
		# return -math.sqrt(
		# 	(self.house.get_rooms(0)[0].get_temp() - self.house.get_rooms(0)[0].get_setpoint()) ** 2 + 
		# 	(self.house.get_rooms(0)[1].get_temp() - self.house.get_rooms(0)[1].get_setpoint()) ** 2
		# )
		reward = -abs(self.house.get_rooms(0)[0].get_temp() - self.house.get_rooms(0)[0].get_setpoint())
		reward -= abs(self.house.get_rooms(0)[1].get_temp() - self.house.get_rooms(0)[1].get_setpoint())
		return reward
	
	def reset(self, seed=None, num_setpoints=1, length=1440, weather_start=None, options=None):
		super().reset()

		self.house = housebuilder.build_house(self._house_cfg)
		self._time = 0
		self._length = length
		self._weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - length) if weather_start is None else weather_start
		self._change_setpoint = set([random.randrange(0, length) for _ in range(num_setpoints - 1)])
		self._ac_last_change = 0
		self._prev_ac = 0
		self._damper0_last_change = 0
		self._prev_damper0 = 0
		self._damper1_last_change = 0
		self._prev_damper1 = 0

		self.house.get_rooms(0)[0].set_setpoint(random.uniform(20, 28))
		self.house.get_rooms(0)[1].set_setpoint(random.uniform(20, 28))

		return self._get_observations(), {}

	def step(self, setting: int):
		power, dampers = self.actions[setting]
		self.house.step(const.OUTSIDE_TEMP[self._weather_start + self._time], power, dampers)
		reward = self._get_reward()
		if power != self._prev_ac:
			self._prev_ac = power
			reward -= 0.65
		if dampers[0][0] != self._prev_damper0:
			self._prev_damper0 = dampers[0][0]
			reward -= 0.1
		if dampers[0][1] != self._prev_damper1:
			self._prev_damper1 = dampers[0][1]
			reward -= 0.1
		terminated = self._time > self._length
		self._time += 1
		return self._get_observations(), reward, terminated, False, {}

	def get_action(self, num: int):
		return self.actions[num]
