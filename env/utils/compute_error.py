import numpy as np
from scipy.spatial.transform import Rotation


def ate_translation(gt_positions, est_positions, mask = None):
    """
    Inputs:
        gt_positions -- groundtruth trajectory (N, L, 3), numpy array type
        est_positions -- estimated trajectory (N, L, 3), numpy array type
        mask -- mask (N, L, 3). Use this to remove some datapoints

        N = number of trajectories
        L = length of trajectories
        3 = x, y, z position

    Output:
        ate_trans -- ate translation (N,)

    """

    r, c, _ = gt_positions.shape
    n = c * np.ones((r, ))

    if isinstance(mask, np.ndarray):
        gt_positions = gt_positions * mask
        est_positions = est_positions * mask
        n = np.sum(mask[:, :, 0], axis=1)

    error_norms = np.linalg.norm(gt_positions-est_positions, axis=2)
    ate_trans = np.sqrt(np.sum(error_norms**2, axis=1) / n)

    return ate_trans


def ate_rotation(gt_rotations, est_rotations, mask = None):
    """
    Inputs:
        gt_positions -- groundtruth quaternions (N, L, 4), numpy array type
        est_positions -- estimated quaternions (N, L, 4), numpy array type
        mask -- mask (N, L, 4). Use this to remove some datapoints

        N = number of trajectories
        L = length of trajectories
        4 = qx, qy, qz, qw quaternion entries

    Output:
        ate_rotation -- ate rotations (N,)

    """

    r, c, _ = gt_rotations.shape
    n = c * np.ones((r, ))

    if isinstance(mask, np.ndarray):
        gt_rotations = gt_rotations * mask
        est_rotations = est_rotations * mask
        # set masked quaternions to identity 
        # we can't keep them as a 0 vector because Rotation.from_quat() will complain
        idx = np.where(np.sum(mask, axis=2) == 0)
        gt_rotations[idx[0], idx[1], 3] = 1.
        est_rotations[idx[0], idx[1], 3] = 1.
        n = np.sum(mask[:, :, 0], axis=1)

    gt_rotations_obj = Rotation.from_quat(gt_rotations.reshape((r*c, 4)))
    est_rotations_obj = Rotation.from_quat(est_rotations.reshape((r*c, 4)))
    e_R = gt_rotations_obj * est_rotations_obj.inv()
    e_v = e_R.as_rotvec().reshape((r, c, 3))
    e_norms = np.linalg.norm(e_v, axis=2)
    ate_rot = np.rad2deg(np.sqrt(np.sum(e_norms**2, axis=1) / n))

    return ate_rot


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1)/2)))*180.0/np.pi


def get_distance_from_start(translation):
    """
    Input:
        translation -- x,y,z values (N, 3), numpy array type
    Output:
        distances -- travelled distances from beginning (N,), numpy array type
    """
    distances = np.diff(translation[:, 0:3], axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances


def compute_comparison_indices_length(distances, dist, max_dist_diff):
    max_idx = len(distances)
    comparisons = []
    for idx, d in enumerate(distances):
        best_idx = -1
        error = max_dist_diff
        for i in range(idx, max_idx):
            if np.abs(distances[i]-(d+dist)) < error:
                best_idx = i
                error = np.abs(distances[i] - (d+dist))
        if best_idx != -1:
            comparisons.append(best_idx)
    return comparisons


def compute_relative_error(
    p_es, q_es, 
    p_gt, q_gt, 
    dist, max_dist_diff,
    scale=1.0,
    T_cm=np.identity(4)):
    """
    Input:
       p_es -- position estimates (N, 3), numpy array type
       q_es -- orientation estimates (N, 4), numpy array type
       p_gt -- groundtruth posiiton estimates (N, 3), numpy array type
       q_gt -- groundtruth orientation estimates (N, 4), numpy array type
       dist -- dimension [m] of the sliding window
       max_dist_diff -- max allowed distance difference [m] wrt the sliding window.
       windows with lengths < max_dist_diff are taken into account. Float value.
       T_cm -- transformation between groundtruth and estimates. Keep to identity. 
       (4, 4), numpy array type
       scale -- Scale alignment. Float value.
    Output:
        error_trans_norm -- In meters (N), numpy array type
        e_rot_abs -- In degrees (N), numpy array type
    """

    accum_distances = get_distance_from_start(p_gt)
    comparisons = compute_comparison_indices_length(
        accum_distances, dist, max_dist_diff)

    n_samples = len(comparisons)
    if n_samples < 2:
        return np.array([]), np.array([])

    T_mc = np.linalg.inv(T_cm)
    errors = []
    for idx, c in enumerate(comparisons):
        if not c == -1:
            T_c1 = np.identity(4)
            T_c1[0:3, 0:3] = Rotation.from_quat(q_es[idx, :]).as_matrix()
            T_c1[0:3, 3] = p_es[idx, :]

            T_c2 = np.identity(4)
            T_c2[0:3, 0:3] = Rotation.from_quat(q_es[c, :]).as_matrix()
            T_c2[0:3, 3] = p_es[c, :]

            T_c1_c2 = np.dot(np.linalg.inv(T_c1), T_c2)
            T_c1_c2[:3, 3] *= scale

            T_m1 = np.identity(4)
            T_m1[0:3, 0:3] = Rotation.from_quat(q_gt[idx, :]).as_matrix()
            T_m1[0:3, 3] = p_gt[idx, :]

            T_m2 = np.identity(4)
            T_m2[0:3, 0:3] = Rotation.from_quat(q_gt[c, :]).as_matrix()
            T_m2[0:3, 3] = p_gt[c, :]

            T_m1_m2 = np.dot(np.linalg.inv(T_m1), T_m2)

            T_m1_m2_in_c1 = np.dot(T_cm, np.dot(T_m1_m2, T_mc))
            T_error_in_c2 = np.dot(np.linalg.inv(T_m1_m2_in_c1), T_c1_c2)
            T_c2_rot = np.eye(4)
            T_c2_rot[0:3, 0:3] = T_c2[0:3, 0:3]
            T_error_in_w = np.dot(T_c2_rot, np.dot(
                T_error_in_c2, np.linalg.inv(T_c2_rot)))
            errors.append(T_error_in_w)

    error_trans_norm = []
    e_rot_abs = []
    for e in errors:
        tn = np.linalg.norm(e[0:3, 3])
        error_trans_norm.append(tn)
        e_rot_abs.append(np.abs(compute_angle(e)))
    return np.array(error_trans_norm), np.array(e_rot_abs)
