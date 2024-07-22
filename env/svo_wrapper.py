import sys
import os
import tqdm
from scipy.spatial.transform import Rotation

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
svo_lib_build_path = os.path.join('svo-lib/build/svo_env')
sys.path.append(svo_lib_build_path)
import svo_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, List, Type
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from dataloader.tartan_loader import TartanLoader
from dataloader.euroc_loader import EurocLoader
from dataloader.tum_loader import TumLoader
from env.utils.trajectory_alignment import align_umeyama
from env.utils.running_mean_std import RunningMeanStd


class VecSVOEnv(VecEnv):
    """Custom Environment that follows gym interface."""
    def __init__(self, params_yaml_path, calib_yaml_path, dataset_dir, num_envs, mode, reward_config,
                 initialize_glog=False, val_traj_ids=None, dataset='tartanair'):
        self.num_envs = num_envs
        self.mode = mode
        self.val_traj_ids = val_traj_ids

        self.reward_coefficients = {
            'align_reward': reward_config.align_reward,
            'keyframe_reward': reward_config.keyframe_reward,
        }
        self.reward_traj_length = reward_config.traj_length
        self.reward_nr_points_for_align = reward_config.nr_points_for_align
        self.reward_scale_length = 5
        self.reward_scale_nr_points = 4
        self.agent_obs_dim_variable = 180*3
        self.agent_obs_dim_fixed = 24
        self.agent_obs_dim = self.agent_obs_dim_variable + self.agent_obs_dim_fixed
        self.critique_dim = 1+6
        self.obs_dim = self.agent_obs_dim + self.critique_dim
        self.action_space = spaces.MultiDiscrete([2, 5])
        self.action_space_scale = np.asarray([[1, 0], [5, 20]])
        self.action_dim = 2
        self.delta_time = float(1/30) * 1e9
        self.step_idx = 0
        self.last_observations = None
        self.observation_space = spaces.Box(np.ones(self.obs_dim) * -np.Inf,
                                            np.ones(self.obs_dim) * np.Inf,
                                            dtype=np.float64)
        self.timestamps = np.zeros([self.num_envs], dtype=np.float64)
        self.env_steps = np.zeros([self.num_envs], dtype='int')
        self.positions = np.zeros([self.num_envs, self.reward_traj_length, 3])
        self.gt_positions = np.zeros([self.num_envs, self.reward_traj_length, 3])
        self.positions_scale = np.zeros([self.num_envs, self.reward_scale_nr_points, 3])
        self.gt_positions_scale = np.zeros([self.num_envs, self.reward_scale_nr_points, 3])
        self.scale_buffer = np.zeros([self.num_envs, self.reward_scale_length])
        self.prev_svo_valid_stage = np.zeros([self.num_envs], dtype='bool')
        self.svo_stages = np.zeros([self.num_envs], dtype='int')

        self.obs_rms = RunningMeanStd(shape=(1, self.agent_obs_dim_fixed))
        self.obs_rms_new = RunningMeanStd(shape=[1, self.agent_obs_dim_fixed])

        self.env = svo_env.SVOEnv(params_yaml_path, calib_yaml_path, num_envs, initialize_glog)

        if dataset == 'tartanair':
            self.dataloader = TartanLoader(dataset_dir, self.mode, self.num_envs, self.val_traj_ids)
        elif dataset == 'euroc':
            self.dataloader = EurocLoader(dataset_dir, self.mode, self.num_envs, self.val_traj_ids)
        elif dataset == 'tum':
            self.dataloader = TumLoader(dataset_dir, self.mode, self.num_envs, self.val_traj_ids)
        self.dataloader_iter = iter(self.dataloader)

        self.timing_dict = None

    def get_images_pose(self):
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        return batch

    def svo_step(self, images, action, timestamps, use_RL_actions, use_gt_init_poses, gt_init_poses=None):
        poses = np.zeros([self.num_envs, 16], dtype=np.float64)
        observations = np.zeros([self.num_envs, self.agent_obs_dim], dtype=np.float64)
        dones = np.zeros([self.num_envs], dtype=np.float64)
        stages = np.zeros([self.num_envs], dtype=np.float64)
        runtime = np.zeros([self.num_envs], dtype=np.float64)

        if not use_gt_init_poses.any():
            gt_init_poses = -np.ones([self.num_envs, 7], dtype=np.float64)

        use_gt_init_poses = use_gt_init_poses.astype(np.float64)
        self.env.step(images, timestamps, action, use_RL_actions, poses, observations, dones, stages, runtime,
                      use_gt_init_poses, gt_init_poses)

        self.svo_stages = stages.astype('int')

        return poses, observations, dones

    def step(self, action, use_RL_actions_bool=True, use_gt_initialization=False):
        action = action.astype(np.float64)
        action = self.action_space_scale[None, :, 0] * action + self.action_space_scale[None, :, 1]

        images, gt_poses, new_seq = self.get_images_pose()
        gt_poses, next_poses = self.extract_next_poses(gt_poses)

        info = [{'new_seq': bool(new_seq[i])} for i in range(self.num_envs)]
        if new_seq.sum() != 0:
            self.reset_new_seq(new_seq, info)

        use_RL_actions, use_gt_init_poses = self.create_options(use_RL_actions_bool, use_gt_initialization)
        poses, observations, svo_dones = self.svo_step(images, action, self.timestamps, use_RL_actions,
                                                       use_gt_init_poses, gt_init_poses=gt_poses)

        # SVO Step
        if svo_dones.sum() > 0:
            poses, observations, info = self.reset_dones(svo_dones, images, action, poses, observations, info,
                                                         use_gt_initialization, gt_init_poses=gt_poses)
        # Also include starts from the dataset
        dones = np.logical_or(svo_dones, new_seq)

        self.step_idx += 1
        self.timestamps += self.delta_time

        svo_valid_stage = self.svo_stages == 2
        valid_stages = np.logical_and(svo_valid_stage, self.prev_svo_valid_stage)

        # Buffer for trajectory alignment
        pos_svo_mask = svo_valid_stage  # Exclude kPause, kInitialization, kRelocalization
        self.update_alignment_buffer(pos_svo_mask, poses, gt_poses)

        reward, info, position_error = self.compute_reward(dones, poses, gt_poses, svo_dones, valid_stages, info, action)

        # Critique observations
        observations = self.filter_observations(svo_valid_stage, observations)
        observations = self.add_critique_observations(observations, gt_poses, next_poses, position_error)
        observations = self.normalize_obs(observations)
        self.last_observations = observations
        self.prev_svo_valid_stage = svo_valid_stage

        if self.mode == 'val':
            for i in range(self.num_envs):
                pred_rot = Rotation.from_matrix(poses[i, :].reshape([-1, 4, 4]).swapaxes(-1, -2)[:, :3, :3])
                info[i]['position'] = poses[i, -4:-1]
                info[i]['rotation'] = pred_rot.as_quat()
                info[i]['gt_position'] = gt_poses[i, :3]
                info[i]['gt_rotation'] = gt_poses[i, 3:]
                info[i]['image'] = images[i, :, :, :]
                info[i]['vo_stages'] = self.svo_stages[i]

        return observations, reward, dones, info, valid_stages

    def reset_dones(self, dones, images, action, poses, observations, info, use_gt_initialization, gt_init_poses=None):
        dones_mask = dones.astype("bool")
        nr_resets = int(dones.sum())
        reset_idx = np.nonzero(dones_mask)[0].astype(np.float64)
        self.env.reset(reset_idx)
        self.timestamps[dones_mask] = 0
        self.env_steps[dones_mask] = 0
        self.positions[dones_mask, :, :] = 0
        self.gt_positions[dones_mask, :, :] = 0
        self.positions_scale[dones_mask, :, :] = 0
        self.gt_positions_scale[dones_mask, :, :] = 0
        self.scale_buffer[dones_mask, :] = 0
        reset_poses = np.zeros([nr_resets, 16], dtype=np.float64)
        reset_observations = np.zeros([nr_resets, self.agent_obs_dim], dtype=np.float64)
        reset_dones_array = np.zeros([nr_resets], dtype=np.float64)
        reset_stages = np.zeros([nr_resets], dtype=np.float64)
        reset_runtime = np.zeros([nr_resets], dtype=np.float64)
        reset_use_RL_actions = np.zeros([nr_resets], dtype=np.float64)
        if not use_gt_initialization:
            use_gt_init_poses = np.zeros([self.num_envs], dtype=np.float64)
            gt_init_poses = -np.ones([self.num_envs, 7], dtype=np.float64)
        else:
            use_gt_init_poses = np.zeros([self.num_envs], dtype=np.float64)
            use_gt_init_poses[dones_mask] = True

        self.env.env_step(reset_idx,
                          images[dones_mask, :, :, :],
                          self.timestamps[dones_mask],
                          action[dones_mask],
                          reset_use_RL_actions,
                          reset_poses,
                          reset_observations,
                          reset_dones_array,
                          reset_stages,
                          reset_runtime,
                          use_gt_init_poses,
                          gt_init_poses)

        poses[dones_mask] = reset_poses
        observations[dones_mask] = reset_observations
        self.svo_stages[dones_mask] = reset_stages.astype('int')

        return poses, observations, info

    def reset(self, seed=None, options=None, use_gt_initialization=False):
        self.env.reset(np.arange(self.num_envs).astype(np.float64))
        if self.mode == 'val':
            self.dataloader.set_up_running_indices()

        images, gt_poses, new_seq = self.get_images_pose()
        gt_poses, next_poses = self.extract_next_poses(gt_poses)

        action = np.zeros([self.num_envs, self.action_dim])
        if use_gt_initialization:
            use_gt_init_poses = np.ones([self.num_envs],  dtype="bool")
        else:
            use_gt_init_poses = np.zeros([self.num_envs], dtype="bool")
        poses, observations, dones = self.svo_step(images, action, self.timestamps,
                                                   use_RL_actions=np.zeros([self.num_envs], dtype=np.float64),
                                                   use_gt_init_poses=use_gt_init_poses,
                                                   gt_init_poses=gt_poses)

        svo_valid_stage = self.svo_stages == 2
        observations = self.filter_observations(svo_valid_stage, observations)
        observations = self.add_critique_observations(observations, gt_poses, next_poses, np.zeros([self.num_envs]))
        observations = self.normalize_obs(observations)

        self.prev_svo_valid_stage = svo_valid_stage
        self.timestamps[:] += self.delta_time
        self.env_steps[:] = 0
        self.positions[:, :, :] = 0
        self.gt_positions[:, :, :] = 0
        self.positions_scale[:, :, :] = 0
        self.gt_positions_scale[:, :, :] = 0
        self.scale_buffer[:, :] = 0

        return observations

    def reset_new_seq(self, new_seq, info):
        new_seq_mask = new_seq.astype("bool")
        nr_resets = int(new_seq.sum())
        reset_idx = np.nonzero(new_seq_mask)[0].astype(np.float64)
        self.env.reset(reset_idx)
        self.timestamps[new_seq_mask] = 0
        self.env_steps[new_seq_mask] = 0
        self.positions[new_seq_mask, :, :] = 0
        self.gt_positions[new_seq_mask, :, :] = 0
        self.positions_scale[new_seq_mask, :, :] = 0
        self.gt_positions_scale[new_seq_mask, :, :] = 0
        self.scale_buffer[new_seq_mask, :] = 0
        self.svo_stages[new_seq] = 1  # SVO state initialization

        for i in range(nr_resets):
            env_id = int(reset_idx[i])
            info[env_id]['terminal_observation'] = self.last_observations[env_id, :]

        return info

    def compute_reward(self, dones, poses, gt_poses, svo_dones, valid_stages, info, action):
        # Position Reward
        pos_rew_mask = np.logical_and(np.logical_not(dones.astype("bool")),
                                      self.env_steps > self.reward_nr_points_for_align)
        position_reward = np.zeros([self.num_envs])
        position_error_array = np.zeros([self.num_envs])
        if int(pos_rew_mask.sum()) != 0:
            s, R, t = align_umeyama(self.gt_positions[pos_rew_mask, :self.reward_nr_points_for_align, :],
                                    self.positions[pos_rew_mask, :self.reward_nr_points_for_align, :])
            # Translation Error
            aligned_pred_position = s[:, None] * np.matmul(R, poses[pos_rew_mask, -4:-1, None]).squeeze(2) + t
            position_error = np.sqrt(((aligned_pred_position - gt_poses[pos_rew_mask, :3])**2).sum(1))
            position_error_array[pos_rew_mask] = position_error

            position_reward[pos_rew_mask] = np.maximum(0.2-position_error, -1) * self.reward_coefficients['align_reward']

        # Keyframe Penalty
        keyframe_reward = -action[:, 0] * self.reward_coefficients['keyframe_reward'] * valid_stages

        reward = position_reward + keyframe_reward

        for i in range(self.num_envs):
            info[i]['position_reward'] = position_reward[i]
            info[i]['keyframe_reward'] = keyframe_reward[i]

        return reward, info, position_error_array

    def update_alignment_buffer(self, pos_svo_mask, poses, gt_poses):
        full_mask = np.logical_and(self.env_steps >= self.reward_traj_length, pos_svo_mask)
        if full_mask.any():
            self.positions[full_mask, :-1, :] = self.positions[full_mask, 1:, :]
            self.gt_positions[full_mask, :-1, :] = self.gt_positions[full_mask, 1:, :]
        if pos_svo_mask.any():
            buffer_idx = np.minimum(self.env_steps[pos_svo_mask], self.reward_traj_length-1)
            self.positions[pos_svo_mask, buffer_idx , :] = poses[pos_svo_mask, -4:-1]
            self.gt_positions[pos_svo_mask, buffer_idx, :] = gt_poses[pos_svo_mask, :3]
            self.env_steps[pos_svo_mask] += 1

            self.positions_scale[pos_svo_mask, 1:, :] = self.positions_scale[pos_svo_mask, :-1, :]
            self.gt_positions_scale[pos_svo_mask, 1:, :] = self.gt_positions_scale[pos_svo_mask, :-1, :]
            self.positions_scale[pos_svo_mask, 0, :] = poses[pos_svo_mask, -4:-1]
            self.gt_positions_scale[pos_svo_mask, 0, :] = gt_poses[pos_svo_mask, :3]

    def create_options(self, use_RL_actions_bool, use_gt_initialization):
        if use_RL_actions_bool:
            use_RL_actions = self.prev_svo_valid_stage.astype(np.float64)
        else:
            use_RL_actions = np.zeros_like(self.prev_svo_valid_stage).astype(np.float64)
        if use_gt_initialization:
            use_gt_init_poses = self.svo_stages == 1
        else:
            use_gt_init_poses = np.zeros([self.num_envs], dtype="bool")

        return use_RL_actions, use_gt_init_poses

    def filter_observations(self, valid_stages, observations):
        filtered_observations = np.zeros([self.num_envs, self.agent_obs_dim], dtype=np.float64)
        filtered_observations[valid_stages, :] = observations[valid_stages, :]

        # Do not filter id since last keyframe
        filtered_observations[:, 1] = observations[:, 1]

        return filtered_observations

    def extract_next_poses(self, gt_poses):
        if self.mode == 'train':
            next_poses = gt_poses[:, 1, :]
            gt_poses = gt_poses[:, 0, :]
        elif gt_poses.ndim == 3:
            gt_poses = gt_poses[:, 0, :]
            next_poses = np.zeros_like(gt_poses)
        else:
            next_poses = np.zeros_like(gt_poses)

        return gt_poses, next_poses

    def add_critique_observations(self, observations, gt_poses, next_poses, position_error):
        critique_obs = np.zeros([self.num_envs, self.critique_dim])
        if self.mode != 'train':
            return np.concatenate([observations, critique_obs], axis=1)

        critique_obs[:, 0] = position_error
        translation = next_poses[:, :3] - gt_poses[:, :3]
        critique_obs[:, 1:4] = translation
        rot_mask = np.abs(translation).sum(1) != 0
        if rot_mask.sum() != 0:
            cur_rotations = Rotation.from_quat(gt_poses[rot_mask, 3:])
            next_rotations = Rotation.from_quat(next_poses[rot_mask, 3:])
            diff_rot = cur_rotations * next_rotations.inv()
            critique_obs[rot_mask, 4:] = diff_rot.as_rotvec()

        return np.concatenate([observations, critique_obs], axis=1)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.mode == 'train':
            self.obs_rms_new.update(obs[:, :self.agent_obs_dim_fixed])

        obs[:, :self.agent_obs_dim_fixed] = self._normalize_obs(obs[:, :self.agent_obs_dim_fixed], self.obs_rms)
        return obs

    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def save_rms(self, data_path) -> None:
        np.savez(
            data_path,
            mean=np.asarray(self.obs_rms.mean),
            var=np.asarray(self.obs_rms.var),
        )

    def load_rms(self, data_dir) -> None:
        self.mean, self.var = None, None
        np_file = np.load(data_dir)

        self.mean = np_file["mean"]
        self.var = np_file["var"]

        self.obs_rms.mean = np.mean(self.mean, axis=0)
        self.obs_rms.var = np.mean(self.var, axis=0)

    def _normalize_obs(self, obs, obs_rms: RunningMeanStd):
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """

        obs[:, :self.agent_obs_dim_fixed] = self._unnormalize_obs(obs[:, :self.agent_obs_dim_fixed], self.obs_rms)
        return obs

    def _unnormalize_obs(self, obs, obs_rms: RunningMeanStd):
        return (np.sqrt(obs_rms.var + 1e-8) * obs) + obs_rms.mean

    def visualize_features(self, env_idx):
        img_shape = (self.dataloader.img_h, self.dataloader.img_w, 3)
        feature_image = np.zeros(img_shape, dtype='uint8')

        self.env.env_visualize_features(env_idx, feature_image, int(self.timestamps[env_idx] - self.delta_time))

        return feature_image

    def render(self):
        raise NotImplementedError

    def close(self):
        del self.env

    def seed(self, seed=0):
        self.env.setSeed(seed)

    def test_dataloader(self):
        test_traj_idx = np.zeros([self.num_envs])
        test_time_idx = np.zeros([self.num_envs])

        for _ in range(3):
            for i_batch in tqdm.tqdm(range(self.dataloader.__len__()), total=self.dataloader.__len__()):
                images, poses, new_seq = self.get_images_pose()
                batch_traj_idx = images[:, 0, 0, 0]
                batch_time_idx = images[:, 0, 0, 1]

                new_seq = new_seq.squeeze()
                not_new = np.logical_not(new_seq)
                if (test_traj_idx[not_new] != batch_traj_idx[not_new]).any():
                    print("Discrepancy detected: 1")
                if (test_time_idx[not_new] + 1 != batch_time_idx[not_new]).any():
                    print("Discrepancy detected: 2")
                if (batch_time_idx[new_seq] != np.zeros([new_seq.sum()])).any():
                    print("Discrepancy detected: 3")

                test_traj_idx[:] = batch_traj_idx
                test_time_idx[:] = batch_time_idx

                if new_seq.sum() == self.num_envs:
                    print("Complete Reset")

    def get_stage(self, stages_int):
        assert stages_int.max() < 4
        assert stages_int.min() >= 0
        svo_stages = [
            'kPaused',  # Stage at the beginning and after reset
            'kInitializing',  # Stage until the first frame with enough features is found
            'kTracking',  # Stage when SVO is running and everything is well
            'kRelocalization',  # Stage when SVO looses tracking and it tries to relocalize
        ]
        return [svo_stages[stage_int] for stage_int in stages_int]


# The following are (not implemented) methods for the abstract parent methods
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        raise NotImplementedError

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise NotImplementedError

    def step_async(self):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError
