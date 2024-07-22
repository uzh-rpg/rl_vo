import os
import cv2
import tqdm
import hydra
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
from omegaconf import OmegaConf


from policies.attention_policy import CustomActorCriticPolicy
from env.svo_wrapper import VecSVOEnv
from env.utils.trajectory_alignment import align_umeyama
from env.utils.visualization import visualize_RL_image, add_text_to_image
from rl_algorithms.buffers import MaskedRolloutBuffer


def evaluation_epoch(val_env, policy, rollout_buffer, test_seq_ids, test_split, config):
    action_space = val_env.action_space
    obs = val_env.reset(use_gt_initialization=config.use_gt_initialization)

    nr_eval_seqs = len(val_env.dataloader.trajectories_paths)
    last_episode_starts = np.ones([nr_eval_seqs])

    completed_bool = np.zeros([nr_eval_seqs], dtype='bool')
    nr_samples_traj = np.zeros([nr_eval_seqs])
    eval_reward = np.zeros([nr_eval_seqs])
    eval_pose = np.zeros([nr_eval_seqs, config.max_eval_steps, 7])
    eval_gt_pose = np.zeros([nr_eval_seqs, config.max_eval_steps, 7])
    eval_valid_stages = np.zeros([nr_eval_seqs, config.max_eval_steps])
    eval_dones = np.zeros([nr_eval_seqs, config.max_eval_steps])
    eval_rewards = np.zeros([nr_eval_seqs, config.max_eval_steps])
    eval_actions = np.zeros([nr_eval_seqs, config.max_eval_steps, 2])
    eval_keyframe_selection = np.zeros([nr_eval_seqs, config.max_eval_steps])

    norm_check_keyframe_dist = np.zeros([nr_eval_seqs])

    for i_eval in tqdm.tqdm(range(config.max_eval_steps)):
        obs_tensor = obs_as_tensor(obs, policy.device)
        with th.no_grad():
            actions, values, log_probs = policy.forward(obs_tensor, deterministic=True)

        clipped_actions = actions.cpu().numpy()
        if isinstance(action_space, spaces.Box):
            if policy.squash_output:
                clipped_actions = policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(actions, action_space.low, action_space.high)

        obs, rewards, dones, infos, valid_mask = val_env.step(clipped_actions,
                                                              use_RL_actions_bool=config.use_rl_actions,
                                                              use_gt_initialization=config.use_gt_initialization)

        rollout_buffer.add(
            obs,  # type: ignore[arg-type]
            actions.cpu().numpy(),
            rewards,
            last_episode_starts,  # type: ignore[arg-type]
            values,
            log_probs,
            valid_mask,
        )
        last_episode_starts = dones

        for i_info, info_dict in enumerate(infos):
            completed_bool[i_info] = np.logical_or(info_dict['new_seq'], completed_bool[i_info])
            if not completed_bool[i_info]:
                eval_pose[i_info, i_eval, :3] = info_dict['position']
                eval_pose[i_info, i_eval, 3:] = info_dict['rotation']
                eval_gt_pose[i_info, i_eval, :3] = info_dict['gt_position']
                eval_gt_pose[i_info, i_eval, 3:] = info_dict['gt_rotation']
                eval_rewards[i_info, i_eval] = rewards[i_info]

        # To unnormalization with check
        unnormalized_obs = obs.copy()
        unnormalized_obs = val_env.unnormalize_obs(unnormalized_obs)
        dist_last_keyframe = np.round(unnormalized_obs[:, 1])
        norm_check_replace = dones if i_eval != 0 else np.ones([nr_eval_seqs], dtype=bool)
        norm_check_keyframe_dist[norm_check_replace] = dist_last_keyframe[norm_check_replace] - 1
        if config.vo_algorithm == "SVO":
            assert ((dist_last_keyframe - unnormalized_obs[:, 1]) < 1e-5).all()
            assert np.logical_or(dist_last_keyframe == norm_check_keyframe_dist + 1,
                                 dist_last_keyframe == 0).all()
        norm_check_keyframe_dist = dist_last_keyframe

        eval_keyframe_selection[:, i_eval] = dist_last_keyframe == 0
        eval_actions[:, i_eval, :] = clipped_actions * valid_mask[:, None] * np.logical_not(completed_bool)[:, None]
        eval_dones[:, i_eval] = dones * np.logical_not(completed_bool)
        eval_valid_stages[:, i_eval] = valid_mask * np.logical_not(completed_bool)
        eval_reward += np.logical_not(completed_bool) * rewards
        nr_samples_traj += np.logical_not(completed_bool)

        if config.visualize_trajs:
            # Save Images
            for i_viz in range(nr_eval_seqs):
                if completed_bool[i_viz]:
                    continue
                viz_log_dir = os.path.join(config.eval_log, '_'.join(test_split[test_seq_ids[i_viz]].split(os.sep)))
                if not os.path.isdir(viz_log_dir):
                    os.makedirs(viz_log_dir)

                if config.use_rl_actions:
                    viz_actions = val_env.action_space_scale[None, :, 0] * actions.cpu().numpy() + val_env.action_space_scale[None, :, 1]
                else:
                    if actions.shape[1] == 2:
                        viz_actions = np.ones([nr_eval_seqs, 2])
                        viz_actions[:, 1] = 20
                    else:
                        viz_actions = np.ones([nr_eval_seqs, 1])
                    viz_actions[:, 0] = unnormalized_obs[:, 1] == 0
                feature_image = visualize_RL_image(val_env, i_viz, infos, valid_mask, rewards, viz_actions, unnormalized_obs,
                                                   # add_actions=config.use_rl_actions)
                                                   add_actions=True)
                file_path = os.path.join(viz_log_dir, 'image_{}.png'.format(i_eval))
                cv2.imwrite(file_path, feature_image)
            # End Save Images

        if np.logical_not(completed_bool).sum() == 0:
            break

    sequence_names = ['_'.join(test_split[test_seq_ids[i_seq]].split(os.sep)) for i_seq in range(nr_eval_seqs)]
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=np.ones_like(dones), last_valid_step=valid_mask)

    print("============ Results ============")
    print("Nr Valid States: {}".format(eval_valid_stages.sum()))
    print("Mean Keyframe  : {}".format(eval_actions[:, :, 0].sum() / eval_valid_stages.sum()))
    print("Total Reward   : {}".format(eval_rewards.sum()))
    print("=================================")

    return {'valid_stages': eval_valid_stages,
            'rewards': eval_rewards,
            'poses': eval_pose,
            'gt_poses': eval_gt_pose,
            'nr_samples_traj': nr_samples_traj,
            'actions': eval_actions,
            'keyframe_selection': eval_keyframe_selection,
            'dones': eval_dones,
            'returns': rollout_buffer.returns,
            'sequence_names': sequence_names}


def visualize_subtrajectories(eval_dict, test_seq_ids, test_split, config):
    new_subtraj = np.cumsum(eval_dict['dones'], axis=1)

    for env_id in range(len(test_seq_ids)):
        nr_subj = int(new_subtraj[env_id, -1]) + 1
        env_pred_positions = np.zeros([config.max_eval_steps, 3])
        for i_subtract in range(nr_subj):
            # subtraj_mask = np.logical_and(eval_dict['valid_stages'][env_id, :], new_subtraj[env_id] == i_subtract)
            subtraj_mask = new_subtraj[env_id] == i_subtract
            if subtraj_mask.sum() < 3:
                continue

            sliced_poses = eval_dict['poses'][env_id, subtraj_mask, :3]
            subtraj_gt = eval_dict['gt_poses'][env_id, subtraj_mask, :3]
            s, R, t = align_umeyama(subtraj_gt[None, :, :], sliced_poses[None, :, :])

            aligned_subtraj = s[:, None, None] * np.matmul(R, sliced_poses[None, :, :, None]).squeeze(3) + t[:, None, :]
            env_pred_positions[subtraj_mask, :] = aligned_subtraj.squeeze(0)

        traj_dir = os.path.join(config.eval_log, '_'.join(test_split[test_seq_ids[env_id]].split(os.sep)))
        for i_time in tqdm.tqdm(range(int(eval_dict['nr_samples_traj'][env_id]))):
            filename = os.path.join(traj_dir, 'image_{}.png'.format(i_time))
            img = cv2.imread(filename)

            img = add_text_to_image(img, "Return: {:.4f}".format(eval_dict['returns'][i_time, env_id]), position='topright')

            # Matplotlib
            fig = plt.figure(layout="constrained", figsize=(10, 10))
            gs = GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])

            # Image with features
            ax1.imshow(img[:, :, ::-1])
            ax1.set_axis_off()

            # Predicted and GT trajectory
            start_idx = max(0, i_time - 50)
            end_idx = i_time + 1
            gt_x = eval_dict['gt_poses'][env_id, start_idx:end_idx, 0]
            gt_y = eval_dict['gt_poses'][env_id, start_idx:end_idx, 1]
            gt_z = eval_dict['gt_poses'][env_id, start_idx:end_idx, 2]
            pred_x = env_pred_positions[start_idx:end_idx, 0]
            pred_y = env_pred_positions[start_idx:end_idx, 1]
            pred_z = env_pred_positions[start_idx:end_idx, 2]
            sliced_keyframe_action = eval_dict['actions'][env_id, start_idx:end_idx, 0]

            for ax, gt_data, pred_data in zip([ax2, ax3],
                                              [[gt_x, gt_z], [gt_x, -gt_y]],
                                              [[pred_x, pred_z], [pred_x, -pred_y]]):
                # Ground Truth
                ax.scatter(gt_data[0] - gt_data[0][-1], gt_data[1] - gt_data[1][-1],
                           c=['dimgray' if subtraj_id % 2 == 0 else 'r' for subtraj_id in new_subtraj[env_id, start_idx:end_idx]])

                # Prediction
                pred_mask = (pred_data[0] * pred_data[1]) != 0
                scat_ax = ax.scatter(pred_data[0][pred_mask] - gt_data[0][-1],
                                     pred_data[1][pred_mask] - gt_data[1][-1],
                                     c=eval_dict['rewards'][env_id, start_idx:end_idx][pred_mask],
                                     marker='x',
                                     cmap='cool',
                                     s=100)

                # Add Action
                valid_action_mask = np.logical_and(pred_mask, sliced_keyframe_action == 1)
                ax.scatter(pred_data[0][valid_action_mask] - gt_data[0][-1],
                           pred_data[1][valid_action_mask] - gt_data[1][-1],
                           c='r',
                           marker='+',
                           s=100)

                ax.axis('equal')
                ax.axis('square')

            fig.colorbar(scat_ax, ax=ax)

            ax2.set_xlim((-5, 5))
            ax2.set_ylim((-5, 5))
            ax3.set_xlim((-5, 5))
            ax3.set_ylim((-5, 5))

            plt.savefig(filename, bbox_inches='tight')
            plt.close()

def save_eval_dict(eval_dict, config):
    save_dir = os.path.join(config.eval_log, 'results')
    os.makedirs(save_dir)

    for i_seq, sequence_name in enumerate(eval_dict['sequence_names']):
        nr_samples = int(eval_dict['nr_samples_traj'][i_seq])
        np.savez_compressed(os.path.join(save_dir, sequence_name + '.npz'),
                            valid_stages=eval_dict['valid_stages'][i_seq, :nr_samples],
                            actions=eval_dict['actions'][i_seq, :nr_samples, :],
                            dones=eval_dict['dones'][i_seq, :nr_samples],
                            poses=eval_dict['poses'][i_seq, :nr_samples, :],
                            gt_poses=eval_dict['gt_poses'][i_seq, :nr_samples, :],
                            keyframe_selection=eval_dict['keyframe_selection'][i_seq, :nr_samples],
                            )

def evaluate(val_env, policy, rollout_buffer, test_seq_ids, test_split, config):
    policy.set_training_mode(False)

    eval_dict = evaluation_epoch(val_env, policy, rollout_buffer, test_seq_ids, test_split, config)

    save_eval_dict(eval_dict, config)

def dummy_lr_fn(val: float):
    def func(_):
        return val
    return func

def save_config(config):
    to_copy_config = config.copy()
    del to_copy_config.agent
    del to_copy_config.wandb_logging
    del to_copy_config.wandb_tag
    del to_copy_config.wandb_group
    del to_copy_config.log_path
    del to_copy_config.n_envs
    del to_copy_config.total_timesteps
    del to_copy_config.val_interval
    with open(os.path.join(config.eval_log, "config.yaml"), "w") as f:
        OmegaConf.save(to_copy_config, f)


def evaluate_policy(weight_path, config):
    test_seq_ids = config.test_seq_ids if config.test_seq_ids != -1 else list(np.arange(config.nr_seqs))
    num_envs = len(test_seq_ids)
    val_env = VecSVOEnv(config.svo_params_file, config.svo_calib_file, config.dataset_dir, num_envs,
                        reward_config=config.agent.reward, mode='val', initialize_glog=True, val_traj_ids=test_seq_ids,
                        dataset=config.dataset)
    test_split = val_env.dataloader.test_split
    config.max_eval_steps = config.max_eval_steps if config.max_eval_steps != -1 else int(val_env.dataloader.nr_samples_per_traj.max())

    encoder_kwargs = dict(
        variable_feature_dim=3,
        obs_dim_variable=val_env.agent_obs_dim_variable,
        obs_dim_fixed=val_env.agent_obs_dim_fixed,
        critique_dim=val_env.critique_dim,
    )
    policy_kwargs = dict(
        encoder_kwargs=encoder_kwargs,
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.0,
    )
    policy = CustomActorCriticPolicy(val_env.observation_space, val_env.action_space, lr_schedule=dummy_lr_fn(1e-4), **policy_kwargs)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    policy = policy.to(device)
    policy = policy.load(weight_path)  # Overwrites network weights even if different dimensions are used
    policy.set_training_mode(False)
    val_env.load_rms(weight_path[:-4] + '_rms.npz')
    print("RMS Loded")

    # Buffer to compute return
    rollout_buffer = MaskedRolloutBuffer(
        config.max_eval_steps,
        val_env.observation_space,  # type: ignore[arg-type]
        val_env.action_space,
        device=policy.device,
        gamma=config.agent.gamma,
        gae_lambda=config.agent.gae_lambda,
        n_envs=num_envs,
    )

    evaluate(val_env, policy, rollout_buffer, test_seq_ids, test_split, config)


@hydra.main(config_path='config', config_name='config_eval', version_base=None)
def main(config):
    os.makedirs(config.eval_log)
    save_config(config)

    if type(config.weight_path) == str:
        evaluate_policy(config.weight_path, config)
    else:
        eval_log_dir = config.eval_log
        for i_policy, weight_path in enumerate(config.weight_path):
            weight_path_split = weight_path.split(os.sep)
            weight_eval_log = os.path.join(eval_log_dir, '__'.join([weight_path_split[-3], weight_path_split[-1]]) + '_{}'.format(i_policy))
            config.eval_log = weight_eval_log
            os.makedirs(config.eval_log)

            evaluate_policy(weight_path, config)


if __name__ == "__main__":
    main()
