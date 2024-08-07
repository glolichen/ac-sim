import gymnasium as gym
import imitation.algorithms.dagger
import gym_environment

import torch

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
import stable_baselines3.common.policies

import tempfile
import numpy as np
import enum
import argparse
from typing import Union, Dict

parser = argparse.ArgumentParser(prog="ImitationLearn")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--model")
parser.add_argument("-t", "--timesteps")

# class FeedForward256Policy(stable_baselines3.common.policies.ActorCriticPolicy):
# 	def __init__(self, *args, **kwargs):
# 		"""Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
# 		super().__init__(*args, **kwargs, net_arch=[64, 128])


def main():
	args = parser.parse_args()
	if args.output == None:
		args.output = "models/dagger_out.zip"
		print("warn: no output passed, default to models/dagger_out.zip")
	if args.timesteps == None:
		args.timesteps = 10000
		print("warn: no timesteps passed, default to 10000")

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
		class Status(enum.Enum):
			NEED_COOL = 0
			WANT_COOL = 1
			NEED_HEAT = 2
			WANT_HEAT = 3
			EQUAL = 4
		def _choose_action(self, obs: Union[np.ndarray, Dict[str, np.ndarray]],) -> int:
			num_rooms = int((len(obs) - 3) / 4)

			epsilon = 0.9
			statuses = []
			need_heat, need_cool = 0, 0
			badness_heat, badness_cool = 0, 0
			min_temp, max_temp = 100000, -100000

			old_ac_status = obs[num_rooms * 2 + 2]
			old_dampers = [[]]
			for i in range(num_rooms):
				old_dampers[0].append(obs[num_rooms * 2 + 4 + i * 2])

			for i in range(num_rooms):
				temp, setp = obs[i * 2], obs[i * 2 + 1]
				min_temp = min(min_temp, temp)
				max_temp = max(max_temp, temp)
				if temp < setp - epsilon:
					statuses.append(self.Status.NEED_HEAT.value)
					badness_heat += abs(temp - setp)
					need_heat += 1
				elif temp < setp:
					statuses.append(self.Status.WANT_HEAT.value)
					badness_heat += abs(temp - setp)
				elif temp > setp + epsilon:
					statuses.append(self.Status.NEED_COOL.value)
					badness_cool += abs(temp - setp)
					need_cool += 1
				elif temp > setp:
					statuses.append(self.Status.WANT_COOL.value)
					badness_cool += abs(temp - setp)
				else:
					statuses.append(self.Status.EQUAL.value)

			outside_temp = obs[num_rooms * 2]
			dampers = [[]]
			if need_heat > need_cool:
				if max_temp >= outside_temp:
					for status in statuses:
						if status == self.Status.NEED_HEAT.value or status == self.Status.WANT_HEAT.value:
							dampers[0].append(False)
						else:
							dampers[0].append(True)
					return env.actions.index((1, dampers))
				else:
					return env.actions.index((0, old_dampers))
			if need_cool > need_heat:
				if min_temp <= outside_temp:
					for status in statuses:
						if status == self.Status.NEED_COOL.value or status == self.Status.WANT_COOL.value:
							dampers[0].append(False)
						else:
							dampers[0].append(True)
					return env.actions.index((-1, dampers))
				else:
					return env.actions.index((0, old_dampers))

			if need_cool > 0 and need_heat > 0:
				if badness_cool > badness_heat:
					if min_temp <= outside_temp:
						for status in statuses:
							if status == self.Status.NEED_COOL.value or status == self.Status.WANT_COOL.value:
								dampers[0].append(False)
							else:
								dampers[0].append(True)
						return env.actions.index((-1, dampers))
					else:
						return env.actions.index((0, old_dampers))
				if badness_heat > badness_cool:
					if max_temp >= outside_temp:
						for status in statuses:
							if status == self.Status.NEED_HEAT.value or status == self.Status.WANT_HEAT.value:
								dampers[0].append(False)
							else:
								dampers[0].append(True)
						return env.actions.index((1, dampers))
					else:
						return env.actions.index((0, old_dampers))
				
			return env.actions.index((old_ac_status, old_dampers))

	stupid = DumbPolicy(env.observation_space, env.action_space)

	rng = np.random.default_rng(0)

	if args.model == None:
		continue_policy = None
	else:
		continue_policy = imitation.policies.base.FeedForward32Policy.load(args.model)

	bc_trainer = imitation.algorithms.bc.BC(
		observation_space=env.observation_space,
		action_space=env.action_space,
		rng=rng,
		device="cuda",
		policy=continue_policy
	)

	with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
		print(tmpdir)
		dagger_trainer = imitation.algorithms.dagger.SimpleDAggerTrainer(
			venv=venv,
			scratch_dir=tmpdir,
			expert_policy=stupid,
			bc_trainer=bc_trainer,
			rng=rng,
		)
		before_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(dagger_trainer.policy, env, 100)
		print("before:", np.mean(before_reward))
		dagger_trainer.train(int(args.timesteps))

	after_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(dagger_trainer.policy, env, 100)
	stupid_reward, _ = stable_baselines3.common.evaluation.evaluate_policy(stupid, env, 100)
	print("before:", np.mean(before_reward))
	print("after:", np.mean(after_reward))
	print("expert:", np.mean(stupid_reward))

	dagger_trainer.policy.save(args.output)

if __name__ == "__main__":
	main()