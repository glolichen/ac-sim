import numpy as np
import random
import gymnasium as gym
import sim
import const

class Environment(gym.Env):
	def _sgn(self, num):
		if self._actions[num] < 0: return -1
		if self._actions[num] > 0: return 1
		return 0
	def __init__(self):
		self.observation_space = gym.spaces.Box(0, 4000, shape=(5,), dtype=float)
		self.action_space = gym.spaces.Discrete(6)
		self._actions = [-1, -0.75, -0.5, -0.25, 0, 1]

	def _get_observations(self):
		return np.array([self._cur_temp, self._cur_setpoint, const.OUTSIDE_TEMP[self._time], self._last_toggle, self._old_power], dtype=np.float32)
	def _get_reward(self):
		return -abs(self._cur_setpoint - self._cur_temp) # lower = better, higher = worse
		# return -math.exp(abs(self._target - self._cur_temp) - 1.7)
	
	def reset(self, num_setpoints=1, length=1440, start_time=0):
		super().reset()

		self._setpoint_list = {}
		self._cur_setpoint = self._setpoint_list[0] = random.uniform(20, 28)
		for i in range(num_setpoints - 1):
			self._setpoint_list[random.randrange(1, length)] = random.uniform(20, 28)

		self._cur_temp = random.uniform(20, 28)
		self._old_power = 0
		self._last_toggle = -1000
		self._time = 0
		self._length = length
		self._start_time = start_time

		return self._get_observations(), self._get_reward()

	def step(self, power):
		if self._time in self._setpoint_list:
			self._cur_setpoint = self._setpoint_list[self._time]
		
		self._cur_temp = sim.calc_next_temp(self._actions[power], self._cur_temp, self._time + self._start_time)

		reward = self._get_reward()
		terminated = self._time > self._length

		if self._sgn(self._old_power) != self._sgn(power):
			# reward -= max(0, self._last_toggle - self._time + 15)
			# self._last_toggle = self._time
			# self._sign_changes += 1
			# if self._sign_changes >= 200:
			reward -= 4

		self._time += 1
		self._old_power = power

		return self._get_observations(), reward, terminated
	
	def get_setpoint(self):
		return self._cur_setpoint
	def get_cur_temp(self):
		return self._cur_temp
	
	def calculate_power(self, action):
		return self._actions[action]
