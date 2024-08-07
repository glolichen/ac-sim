import numpy as np
import random
import gymnasium as gym
import housebuilder
import itertools
import const

class Environment(gym.Env):
	num_rooms = 5
	# num_rooms = 2
	def __init__(self):
		self.observation_space = gym.spaces.Box(-10, 3000, shape=(self.num_rooms * 4 + 3,), dtype=float)
		combinations = itertools.product([True, False], repeat=self.num_rooms)
		self.actions = []
		for c in combinations:
			self.actions.append((-1, [list(c)] ))
			self.actions.append(( 0, [list(c)] ))
			self.actions.append(( 1, [list(c)] ))
		self.action_space = gym.spaces.Discrete(len(self.actions))
		self._house_cfg = "5r_crazy.json"
		# self._house_cfg = "2r_simple.json"

	def _get_observations(self):
		ret = np.array([])
		for room in self.house.get_rooms(0):
			ret = np.concatenate((ret, np.array([room.get_temp(), room.get_setpoint()])))
		# num_rooms * 2
		ret = np.concatenate((ret, np.array([
			const.OUTSIDE_TEMP[self._weather_start + self._time],
			self._ac_cycles,
			self._prev_ac
		])))
		# num_rooms * 2 + 3
		for i in range(self.num_rooms):
			ret = np.concatenate((ret, np.array([self._damper_cycles[i], self._prev_damper[i]])))
		return ret
	
	def _get_reward(self):
		reward = 0
		for room in self.house.get_rooms(0):
			reward -= abs(room.get_temp() - room.get_setpoint())
		return reward
	
	def reset(self, seed=None, num_setpoints=1, length=1440, weather_start=None, options=None):
		super().reset()

		self.house = housebuilder.build_house(self._house_cfg)
		self._time = 0
		self._length = length
		self._weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - length) if weather_start is None else weather_start
		self._change_setpoint = set([random.randrange(0, length) for _ in range(num_setpoints - 1)])
		self._ac_cycles = 0
		self._prev_ac = 0

		self._damper_cycles = np.zeros(self.num_rooms)
		self._prev_damper = np.zeros(self.num_rooms)
		for room in self.house.get_rooms(0):
			room.set_setpoint(random.uniform(20, 28))

		return self._get_observations(), {}

	def step(self, setting: int):
		power, dampers = self.actions[setting]
		self.house.step(const.OUTSIDE_TEMP[self._weather_start + self._time], power, dampers)

		reward = self._get_reward()
		if power != self._prev_ac:
			self._prev_ac = power
			self._ac_cycles += 1
			if self._ac_cycles > 130:
				reward -= 2
		
		for i in range(self.num_rooms):
			if dampers[0][i] != self._prev_damper[i]:
				self._prev_damper[i] = dampers[0][i]
				self._damper_cycles[i] += 1
				if self._damper_cycles[i] > 80:
					reward -= 1

		terminated = self._time > self._length
		self._time += 1
		return self._get_observations(), reward, terminated, False, {}

	def get_action(self, num: int):
		return self.actions[num]
