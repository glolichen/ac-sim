import numpy as np
import random
import gymnasium as gym
import sim
import torch
import const

class Environment(gym.Env):
	def _sgn(self, num):
		if self._actions[num] < 0: return -1
		if self._actions[num] > 0: return 1
		return 0
	def __init__(self):
		self.observation_space = gym.spaces.Box(0, 4000, shape=(5,), dtype=float)
		self.action_space = gym.spaces.Discrete(3)
		self._actions = torch.tensor([-1, 0, 1], dtype=torch.long)

	def _get_observations(self):
		return torch.transpose(torch.stack((
			self._cur_temp,
			self._cur_setpoint,
			torch.full((self._concurrent,), const.OUTSIDE_TEMP[self._time]),
			self._last_toggle,
			self._old_power
		)), 0, 1)
	
	def _get_reward(self):
		return -torch.abs(self._cur_setpoint - self._cur_temp) # lower = better, higher = worse
		# return -math.exp(abs(self._target - self._cur_temp) - 1.7)
	
	def reset(self, num_setpoints=1, length=1440, start_time=0, concurrent=8):
		super().reset()

		self._concurrent = concurrent
		self._setpoint_list = {}
		self._cur_setpoint = self._setpoint_list[0] = torch.rand(self._concurrent) * 8 + 20
		for i in range(num_setpoints - 1):
			self._setpoint_list[random.randrange(1, length)] = torch.rand(self._concurrent) * 8 + 20
		
		self._cur_temp = torch.rand(self._concurrent) * 8 + 20
		self._old_power = torch.full((self._concurrent,), 0)
		self._last_toggle = torch.full((self._concurrent,), -1000)
		self._time = 0
		self._length = length
		self._start_time = start_time

		self._all_zeros = torch.zeros(self._concurrent)

		return self._get_observations(), self._get_reward()

	def step(self, power):
		# print(power)
		if self._time in self._setpoint_list:
			self._cur_setpoint = self._setpoint_list[self._time]
		
		self._cur_temp = sim.calc_next_temp(
			torch.take(self._actions, power),
			self._cur_temp,
			self._time + self._start_time
		)

		reward = self._get_reward()
		terminated = self._time > self._length

		unequal_mask = self._old_power != power
		penalty = torch.max(self._all_zeros, self._last_toggle - self._time + 31.5)
		penalty = torch.where(unequal_mask, penalty, torch.tensor(0))
		reward -= penalty

		self._last_toggle = torch.where(unequal_mask, torch.full((self._concurrent,), self._time), torch.tensor(0))

		# if self._sgn(self._old_power) != self._sgn(power):
			# self._last_toggle = self._time
			# self._sign_changes += 1
			# if self._sign_changes >= 200:
			# reward -= 4
			# if self._sgn(self._old_power) != 0 and self._sgn(power) != 0:
			# 	reward -= 100000

		self._time += 1
		self._old_power = power
		
		return self._get_observations(), reward, terminated
	
	def get_setpoint(self):
		return self._cur_setpoint
	def get_cur_temp(self):
		return self._cur_temp
	
	def calculate_power(self, action):
		return self._actions[action]
