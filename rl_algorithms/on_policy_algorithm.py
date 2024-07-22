import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import plotly.express as px
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from rl_algorithms.buffers import MaskedRolloutBuffer
from env.utils.trajectory_alignment import align_umeyama
from env.utils.compute_error import ate_translation

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else MaskedRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos, valid_mask = env.step(clipped_actions, use_gt_initialization=True)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                valid_mask,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones, last_valid_step=valid_mask)

        callback.update_locals(locals())

        callback.on_rollout_end()

        if self.wandb_logging:
            nr_valid_states = rollout_buffer.valid_mask.sum()
            wandb_dir = {
                "rollout/sum_reward": rollout_buffer.rewards.sum(),
                "rollout/sum_valid_stages": nr_valid_states,
                "rollout/ratio_valid_stages": nr_valid_states / float(rollout_buffer.valid_mask.size),
                "rollout/mean_keyframes": (rollout_buffer.actions[:, :, 0] * rollout_buffer.valid_mask[:, :]).sum() / nr_valid_states,
                "rollout/iteration": self.iteration,
            }
            self.wandb_run.log(wandb_dir)


        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        eval_interval: int = -1,
        val_env: Union[GymEnv, str] = None,
    ) -> SelfOnPolicyAlgorithm:
        self.iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            self.iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.iteration % 10 == 0:
                self.env.update_rms()
                val_env.obs_rms = self.env.obs_rms

            print("Iteration:      {}".format(self.iteration))
            print("Total Timestep: {}".format(self.num_timesteps))
            print("==========================")

            # Display training infos
            if log_interval is not None and self.iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", self.iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

            if eval_interval != -1 and self.iteration % eval_interval == 0:
                self.evaluation_epoch(val_env)

                if self.log_dir is not None:
                    policy_path = self.log_dir + "/Policy/"
                    os.makedirs(policy_path, exist_ok=True)
                    # save the model parameters
                    self.policy.save(policy_path + "iter_{0:05d}.pth".format(self.iteration))
                    self.env.save_rms(policy_path + "iter_{0:05d}_rms.npz".format(self.iteration))

        callback.on_training_end()

        return self

    def evaluation_epoch(self, val_env):
        self.policy.set_training_mode(False)
        max_eval_steps = 1500  # SVO
        nr_eval_seqs = len(val_env.dataloader.trajectories_paths)

        completed_bool = np.zeros([nr_eval_seqs], dtype='bool')
        nr_samples_traj = np.zeros([nr_eval_seqs])
        eval_reward = np.zeros([nr_eval_seqs])
        eval_valid_stages = np.zeros([nr_eval_seqs, max_eval_steps])
        eval_dones = np.zeros([nr_eval_seqs, max_eval_steps])
        eval_rewards_dict = {}
        eval_action = np.zeros([nr_eval_seqs, max_eval_steps, self.env.action_dim])
        eval_positions = np.zeros([nr_eval_seqs, max_eval_steps, 3])
        eval_gt_positions = np.zeros([nr_eval_seqs, max_eval_steps, 3])

        obs = val_env.reset()

        for i_eval in range(max_eval_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            with th.no_grad():
                actions, values, log_probs = self.policy.forward(obs_tensor, deterministic=True)

            clipped_actions = actions.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            obs, rewards, dones, infos, valid_mask = val_env.step(clipped_actions, use_gt_initialization=True)

            for i_envs, info_dict in enumerate(infos):
                completed_bool[i_envs] = np.logical_or(info_dict['new_seq'], completed_bool[i_envs])
                if completed_bool[i_envs]:
                    continue
                for key, value in info_dict.items():
                    if 'reward' in key:
                        if key not in eval_rewards_dict:
                            eval_rewards_dict[key] = np.zeros([nr_eval_seqs])
                        eval_rewards_dict[key][i_envs] += value
                eval_positions[i_envs, i_eval, :] = info_dict['position'] * valid_mask[i_envs]
                eval_gt_positions[i_envs, i_eval, :] = info_dict['gt_position'] * valid_mask[i_envs]

            eval_reward += np.logical_not(completed_bool) * rewards
            eval_action[:, i_eval, :] = np.logical_not(completed_bool)[:, None] * actions[:, :].cpu().numpy() * valid_mask[:, None]
            eval_valid_stages[:, i_eval] = np.logical_not(completed_bool) * valid_mask
            eval_dones[:, i_eval] = np.logical_not(completed_bool) * dones
            nr_samples_traj += np.logical_not(completed_bool)

            if np.logical_not(completed_bool).sum() == 0:
                break


        # Compute subtrajs
        new_subtraj = np.cumsum(eval_dones, axis=1)

        # Compute statistics for first subtraj
        first_subtraj_mask = new_subtraj == 0
        ate_traj = np.zeros([nr_eval_seqs])
        for i_traj in range(nr_eval_seqs):
            traj_mask = np.logical_and(first_subtraj_mask[i_traj, :], eval_positions[i_traj, :, :].sum(1) != 0)
            if traj_mask.sum() <= 3:
                continue
            s, R, t = align_umeyama(eval_gt_positions[None, i_traj, traj_mask, :],
                                    eval_positions[None, i_traj, traj_mask, :])
            aligned_position = s[0] * np.matmul(R[0, :, :], eval_positions[i_traj, traj_mask, :, None]).squeeze(2) + t[0]
            ate_traj[i_traj] = ate_translation(eval_gt_positions[None, i_traj, traj_mask, :],
                                               aligned_position[None, :, :])

        # Wandb Logging
        summed_reward_traj = eval_reward
        valid_stages_traj = eval_valid_stages.sum(axis=1)
        nr_subtraj_traj = new_subtraj[:, -1]
        mean_length_subtraj_traj = eval_valid_stages.sum(axis=1) / (nr_subtraj_traj + 1e-9)
        mean_keyframe_traj = eval_action[:, :, 0].sum(axis=1) / (valid_stages_traj + 1e-9)

        if self.wandb_logging:
            wandb_dir = {
                "eval/mean_summed_reward": eval_reward.mean(),
                "eval/sum_valid_stages": eval_valid_stages.sum(),
                "eval/ratio_valid_stages": eval_valid_stages.sum() / nr_samples_traj.sum(),
                "eval/mean_keyframes": mean_keyframe_traj.mean(),
                "eval/mean_ate": ate_traj.mean(),
                "eval/mean_first_seq_valid_stages": first_subtraj_mask.sum(1).mean(),
                "eval/summed_reward_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=summed_reward_traj),
                "eval/mean_length_subtraj_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=mean_length_subtraj_traj),
                "eval/valid_stages_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=valid_stages_traj),
                "eval/mean_keyframe_action_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=mean_keyframe_traj),
                "eval/mean_ate_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=ate_traj),
                "eval/first_seq_valid_stages_per_traj": px.bar(x=np.arange(nr_eval_seqs), y=first_subtraj_mask.sum(1)),
                "eval/iteration": self.iteration,
            }
            for i_dict, (key, value) in enumerate(eval_rewards_dict.items()):
                wandb_dir["eval/mean_" + key] = (value / (nr_samples_traj + 1e-9)).mean()
            self.wandb_run.log(wandb_dir)


    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
