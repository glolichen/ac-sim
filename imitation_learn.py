import gymnasium as gym
import gym_environment

import imitation

import imitation.algorithms

import imitation.data
import imitation.data.rollout
import imitation.data.wrappers

import imitation.policies
import imitation.policies.base

import imitation.util.util

import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.evaluation

import sys

from typing import Union, Dict

import abc
import numpy as np

gym.register(
	id="HVAC-v0",
	entry_point=gym_environment.Environment,
	max_episode_steps=1440,
)

env = gym.make("HVAC-v0")
venv = imitation.util.util.make_vec_env(
	"HVAC-v0",
	rng=np.random.default_rng(),
	n_envs=4,
	post_wrappers=[lambda env, _: imitation.data.wrappers.RolloutInfoWrapper(env)],
)

class DumbPolicy(imitation.policies.base.NonTrainablePolicy):
	def _choose_action(self, obs: Union[np.ndarray, Dict[str, np.ndarray]],) -> int:
		epsilon = 0.9
		
		room0_temp, room0_setp, room1_temp, room1_setp, outside_temp, prev_setting, _ = obs
		prev_ac_status, prev_dampers = env.actions[int(prev_setting)]

		if room0_temp > room0_setp + epsilon and room1_temp > room1_setp:
			ac_status, dampers = (-1, [[False, False]])
		elif room0_temp < room0_setp - epsilon and room1_temp < room1_setp:
			ac_status, dampers = (1, [[False, False]])
		
		elif room0_temp > room0_setp and room1_temp > room1_setp + epsilon:
			ac_status, dampers = (-1, [[False, False]])
		elif room0_temp < room0_setp and room1_temp < room1_setp - epsilon:
			ac_status, dampers = (1, [[False, False]])

		elif room0_temp > room0_setp + epsilon and abs(room1_temp - room1_setp) < epsilon:
			ac_status, dampers = (-1, [[False, True]])
		elif room1_temp > room1_setp + epsilon and abs(room0_temp - room0_setp) < epsilon:
			ac_status, dampers = (-1, [[True, False]])
		
		elif room0_temp < room0_setp - epsilon and abs(room1_temp - room1_setp) < epsilon:
			ac_status, dampers = (1, [[False, True]])
		elif room1_temp < room1_setp - epsilon and abs(room0_temp - room0_setp) < epsilon:
			ac_status, dampers = (1, [[True, False]])
		
		elif room0_temp > room0_setp + epsilon and room1_temp < room1_setp - epsilon:
			error0, error1 = abs(room0_temp - room0_setp), abs(room1_temp - room1_setp)
			if error0 > error1:
				ac_status, dampers = (-1, [[False, True]])
			else:
				ac_status, dampers = (1, [[True, False]])
		elif room0_temp < room0_setp - epsilon and room1_temp > room1_setp + epsilon:
			error0, error1 = abs(room0_temp - room0_setp), abs(room1_temp - room1_setp)
			if error0 > error1:
				ac_status, dampers = (1, [[False, True]])
			else:
				ac_status, dampers = (-1, [[True, False]])
		else:
			ac_status, dampers = (prev_ac_status, prev_dampers)
		
		if room0_temp > outside_temp and room1_temp > outside_temp and ac_status < 0:
			ac_status = 0
			dampers = prev_dampers
		if room0_temp < outside_temp and room1_temp < outside_temp and ac_status > 0:
			ac_status = 0
			dampers = prev_dampers
		
		print(room0_temp, room0_setp, room1_temp, room1_setp, outside_temp, ac_status, dampers)
		return env.actions.index((ac_status, dampers))
	
reward, _ = stable_baselines3.common.evaluation.evaluate_policy(DumbPolicy(env.observation_space, env.action_space), env, 1)
np.set_printoptions(threshold=np.inf)
print(reward)

# stupid = DumbPolicy(env.observation_space, env.action_space)
# rng = np.random.default_rng()
# rollouts = imitation.data.rollout.rollout(
# 	stupid,
# 	venv,
# 	imitation.data.rollout.make_sample_until(min_timesteps=None, min_episodes=1),
# 	rng=rng,
# )
# transitions = imitation.data.rollout.flatten_trajectories(rollouts)

# from imitation.algorithms import bc
# bc_trainer = bc.BC(
# 	observation_space=env.observation_space,
# 	action_space=env.action_space,
# 	demonstrations=transitions,
# 	rng=rng,
# )

# reward_before_training, _ = stable_baselines3.common.evaluation.evaluate_policy(bc_trainer.policy, env, 10)
# print(f"Reward before training: {reward_before_training}")

# bc_trainer.train(n_epochs=1)
# reward_after_training, _ = stable_baselines3.common.evaluation.evaluate_policy(bc_trainer.policy, env, 10)
# print(f"Reward after training: {reward_after_training}")
