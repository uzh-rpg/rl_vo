import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


from env.utils.trajectory_alignment import align_umeyama
from env.utils.compute_error import ate_translation, ate_rotation

METHODS = {
    'with_RL': <path_to_dir>,
    'wo_RL': <path_to_dir>,
}
VISUALIZE_RESULTS= False
OUT_DIR = '/logs/log_voRL/playground/Analysis'


def compute_errors(gt_poses, poses):
    # Alignment
    s, R, t = align_umeyama(gt_poses[None, :, :3], poses[None, :, :3])
    aligned_positions = s[:, None, None] * np.matmul(R, poses[None, :, :3, None]).squeeze(3) + t[:, None, :]
    rotations_matrix = Rotation.from_quat(poses[:, 3:]).as_matrix()
    aligned_rotations = np.matmul(R, rotations_matrix)
    aligned_rotations = Rotation.from_matrix(aligned_rotations).as_quat()

    ate_transl = ate_translation(gt_poses[None, :, :3], aligned_positions)
    ate_rot = ate_rotation(gt_poses[None, :, 3:], aligned_rotations[None, :, :])

    return ate_transl, ate_rot


def evaluate_policy(method_dict):
    sequences = None
    for method, dir_results in method_dict.items():
        method_sequences = [dir_name[:-4] for dir_name in os.listdir(os.path.join(dir_results, 'results'))]
        method_sequences.sort()

        if sequences is None:
            sequences = method_sequences
        else:
            assert sequences==method_sequences

    n_cols = min(len(sequences), 4)
    n_rows = int(np.ceil(len(sequences) / 4))
    if VISUALIZE_RESULTS:
        seq_fig, seq_axs = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(n_cols*5, n_rows*5))

    results_mean_array = np.zeros([len(sequences), len(method_dict.keys()), 4])
    results_min_traj_array = np.zeros([len(sequences), len(method_dict.keys()), 4])
    nr_finished_seqs = np.zeros([len(sequences), len(method_dict.keys())])
    for i_seq, sequence in enumerate(sequences):
        gt_poses = None
        ground_truth_added = False

        min_traj_mask = None
        # Find the smallest trajectory
        for i_method, (method, dir_results) in enumerate(method_dict.items()):
            method_results = np.load(os.path.join(dir_results, 'results', sequence + '.npz'))
            first_mask = np.cumsum(method_results['dones'], axis=0) == 0
            if min_traj_mask is None:
                min_traj_mask = first_mask
            elif min_traj_mask.sum() > first_mask.sum():
                min_traj_mask = first_mask

        if VISUALIZE_RESULTS:
            seq_plot_x = seq_axs[i_seq // n_cols, i_seq % n_cols]

        for i_method, (method, dir_results) in enumerate(method_dict.items()):
            method_results = np.load(os.path.join(dir_results, 'results', sequence + '.npz'))
            new_subtraj = np.cumsum(method_results['dones'], axis=0)

            if sequence == 'rgbd_dataset_freiburg1_floor':
                first_mask = new_subtraj == 0
                invalid_gt_mask = (method_results['gt_poses'] == -1).sum(1) == 7
                first_mask[invalid_gt_mask] = False
            else:
                first_mask = new_subtraj == 0

            method_poses = method_results['poses'][first_mask, :]
            if gt_poses is None:
                gt_poses = method_results['gt_poses']
            else:
                assert not (np.abs(gt_poses - method_results['gt_poses']) > 1e-4).any()

            nr_finished_seqs[i_seq, i_method] += 1 if method_results['gt_poses'].shape[0] == first_mask.sum() else 0

            ate_transl, ate_rot = compute_errors(gt_poses[first_mask, :], method_poses)
            ate_transl_min_traj, ate_rot_min_traj = compute_errors(gt_poses[min_traj_mask, :],
                                                                   method_results['poses'][min_traj_mask, :])

            if np.isnan(ate_transl):
                continue

            results_mean_array[i_seq, i_method, 0] = ate_transl
            results_mean_array[i_seq, i_method, 1] = ate_rot
            results_mean_array[i_seq, i_method, 2] = first_mask.sum()
            results_mean_array[i_seq, i_method, 3] = 1

            results_min_traj_array[i_seq, i_method, 0] = ate_transl_min_traj
            results_min_traj_array[i_seq, i_method, 1] = ate_rot_min_traj
            results_min_traj_array[i_seq, i_method, 2] = min_traj_mask.sum()
            results_min_traj_array[i_seq, i_method, 3] = 1

            if VISUALIZE_RESULTS:
                # Trajectory Plot
                s, R, t = align_umeyama(gt_poses[None, first_mask, :3], method_poses[None, :, :3])
                aligned_positions = s[:, None, None] * np.matmul(R, method_poses[None, :, :3, None]).squeeze(3) + t[:, None, :]
                aligned_positions = aligned_positions.squeeze(0)

                if not ground_truth_added:
                    seq_plot_x.scatter(gt_poses[:, 0], gt_poses[:, 1], s=1, label="GT")
                    seq_plot_x.axis('equal')
                    seq_plot_x.axis('square')
                    seq_plot_x.set_xlabel('x [m]')
                    seq_plot_x.set_ylabel('y [m]')
                    seq_plot_x.set_title(sequence)
                    ground_truth_added = True

                seq_plot_x.scatter(aligned_positions[:, 0], aligned_positions[:, 1], s=1, label=method)
                seq_plot_x.legend()
                seq_plot_x.xaxis.set_major_locator(plt.MaxNLocator(5))
                seq_plot_x.yaxis.set_major_locator(plt.MaxNLocator(5))

            key_frame_selection = method_results['keyframe_selection'][first_mask]
            print("Method: {}, Number of Keyframes: {}".format(method, key_frame_selection.sum()))

    # Print finished sequences
    for i_method, method in enumerate(method_dict.keys()):
        print("Method: {}".format(method))
        print("Number of successful trajectory {}/{}".format(nr_finished_seqs[:, i_method].sum(), nr_finished_seqs.shape[0]))
        print("Not successful: ")
        print([seq for i_seq, seq in enumerate(sequences) if not nr_finished_seqs[i_seq, i_method]])
        print("---------------------------------------")

    if VISUALIZE_RESULTS:
        seq_fig.savefig(os.path.join(OUT_DIR, 'top_down.png'), bbox_inches='tight', dpi=200)
        plt.close()

    results_dict = {'overall': {}, 'intersection': {}}
    valid_run_mask = results_min_traj_array[:, :, 3].sum(1) == len(method_dict.keys())
    valid_runs = valid_run_mask.sum()
    for i_method, method in enumerate(method_dict.keys()):
        valid_runs_method = results_mean_array[:, i_method, 3].sum()
        results_dict['overall'][method] = compute_stats_dict(results_mean_array[:, i_method, :], valid_runs_method)
        results_dict['intersection'][method] = compute_stats_dict(results_min_traj_array[valid_run_mask, i_method, :],
                                                                  valid_runs)
        for i_seq, sequence in enumerate(sequences):
            if i_method == 0:
                results_dict['overall ' + sequence] = {}
                results_dict['intersection ' + sequence] = {}
            results_dict['overall ' + sequence][method] = compute_stats_dict(results_mean_array[i_seq, i_method, :])
            results_dict['intersection ' + sequence][method] = compute_stats_dict(results_min_traj_array[i_seq, i_method, :])

    return results_dict

def compute_stats_dict(result_array, valid_runs=None):
    if valid_runs is not None:
        assert result_array.ndim == 2
        stats_dict = {
            'mean_ate_translation': np.round(result_array[:, 0].sum() / valid_runs, decimals=4),
            'mean_ate_rotation': np.round(result_array[:, 1].sum() / valid_runs, decimals=4),
            'valid_states': result_array[:, 2].sum(),
            'valid_runs': valid_runs,
        }
    else:
        assert result_array.ndim == 1
        stats_dict = {
            'mean_ate_translation': np.round(result_array[0], decimals=4),
            'mean_ate_rotation': np.round(result_array[1], decimals=4),
            'valid_states': result_array[2],
        }

    return stats_dict


def save_to_csv(rl_polices, policy_results):
    with open(os.path.join(OUT_DIR, 'results.csv'), 'w') as f:
        writer = csv.writer(f)
        for i_policy, rl_policy in enumerate(rl_polices):
            writer.writerow([rl_policy])
            policy_result = policy_results[i_policy]

            new_policy_result = {}
            for k, v in policy_result.items():
                if 'intersection' not in k:
                    new_policy_result[k] = v
            policy_result = new_policy_result



            n_rows, n_cols = 4, 2 + len(policy_result.keys())
            rows = [['' for _ in range(n_cols)] for _ in range(n_rows)]

            for i_eval, evaluation_key in enumerate(policy_result.keys()):
                eval_col =  2 + i_eval
                rows[0][eval_col] = evaluation_key
                for i_method, method_key in enumerate(policy_result[evaluation_key]):
                    method_row = 2 + i_method
                    if i_eval == 0:
                        rows[method_row][eval_col - 1] = method_key
                    method_results = policy_result[evaluation_key][method_key]
                    for i_result, (k, v) in enumerate(method_results.items()):
                        if k != 'mean_ate_translation':
                            continue
                        result_col = eval_col + i_result
                        if rows[1][result_col] != '':
                            assert rows[1][result_col] == k
                        else:
                            rows[1][result_col] = k
                        rows[method_row][result_col] = v

            writer.writerows(rows)



def main():
    dir_content = os.listdir(METHODS['with_RL'])
    if 'results' in dir_content:
        policy_results = evaluate_policy(METHODS)
        save_to_csv([METHODS['with_RL'].split(os.sep)[-1]], [policy_results])
    else:
        rl_polices = [policy_dir for policy_dir in dir_content if 'config.yaml' != policy_dir]
        rl_polices.sort()
        policy_results = []
        for rl_policy in rl_polices:
            print(rl_policy)
            cur_method_dict = {'with_RL': os.path.join(METHODS['with_RL'], rl_policy),
                               'wo_RL': METHODS['wo_RL']}
            policy_results.append(evaluate_policy(cur_method_dict))

        save_to_csv(rl_polices, policy_results)


if __name__ == "__main__":
    main()
