import gymnasium as gym
import gym_environment

import imitation

import imitation.data
import imitation.data.rollout
import imitation.data.wrappers

import imitation.policies
import imitation.policies.base

import imitation.algorithms
import imitation.algorithms.adversarial
import imitation.algorithms.adversarial.airl
import imitation.algorithms.adversarial.gail

import imitation.rewards
import imitation.rewards.reward_nets

import imitation.util
import imitation.util.util
import imitation.util.networks

import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.evaluation
import stable_baselines3.ppo

import numpy as np
from typing import Union, Dict

DATASET_SIZE = 20_000
TRAIN_TIMESTEPS = 32_000_000

gym.register(
	id="HVAC-v0",
	entry_point=gym_environment.Environment,
	max_episode_steps=1440,
)

env = gym.make("HVAC-v0")
venv = imitation.util.util.make_vec_env(
	"HVAC-v0",
	rng=np.random.default_rng(),
	n_envs=1,
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
		
		# print(room0_temp, room0_setp, room1_temp, room1_setp, outside_temp, ac_status, dampers)
		return env.actions.index((ac_status, dampers))
	
stupid = DumbPolicy(env.observation_space, env.action_space)
rng = np.random.default_rng()
rollouts = imitation.data.rollout.rollout(
	stupid,
	venv,
	imitation.data.rollout.make_sample_until(min_timesteps=None, min_episodes=DATASET_SIZE),
	rng=rng,
	verbose=True
)
transitions = imitation.data.rollout.flatten_trajectories(rollouts)

print("Finished generating trajectories")

learner = stable_baselines3.PPO(
  env=env,
	policy=stable_baselines3.ppo.MlpPolicy,
	batch_size=64,
	ent_coef=0.0,
	learning_rate=0.0005,
	gamma=0.95,
	clip_range=0.1,
	vf_coef=0.1,
	n_epochs=5
)
# learner = stable_baselines3.PPO.load("imitation_in.zip")

reward_net = imitation.rewards.reward_nets.BasicShapedRewardNet(
	observation_space=env.observation_space,
	action_space=env.action_space,
	normalize_input_layer=imitation.util.networks.RunningNorm,
)
airl_trainer = imitation.algorithms.adversarial.airl.AIRL(
	demonstrations=rollouts,
	demo_batch_size=2048,
	gen_replay_buffer_capacity=512,
	n_disc_updates_per_round=16,
	venv=venv,
	gen_algo=learner,
	reward_net=reward_net,
)

learner_rewards_before_training, _ = stable_baselines3.common.evaluation.evaluate_policy(
	learner, env, 100, return_episode_rewards=True,
)

airl_trainer.train(TRAIN_TIMESTEPS)

learner_rewards_after_training, _ = stable_baselines3.common.evaluation.evaluate_policy(
	learner, env, 100, return_episode_rewards=True,
)
print("before", np.mean(learner_rewards_before_training))
print("after", np.mean(learner_rewards_after_training))

learner.save("imitation_out.zip")
