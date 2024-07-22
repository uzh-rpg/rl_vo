"""Adapted from: https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/align_trajectory.py"""
import numpy as np


def align_umeyama(model, data):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (N, L, 3), numpy array type
    data -- second trajectory (N, L, 3), numpy array type

    Output:
    s -- scale factor (N)
    R -- rotation matrix (N, 3, 3)
    t -- translation vector (N, 3, 1)

    """
    N, l = model.shape[:2]

    # substract mean
    mu_M = model.mean(axis=1, keepdims=True)
    mu_D = data.mean(axis=1, keepdims=True)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D

    # correlation
    C = 1.0/l*np.matmul(np.transpose(model_zerocentered, [0, 2, 1]), data_zerocentered)
    sigma2 = 1.0/l*(data_zerocentered**2).sum(axis=(1, 2))
    U_svd, D_svd, V_svd = np.linalg.svd(C)
    D_diag = np.zeros([N, 3, 3])
    D_diag[:, [0, 1, 2], [0, 1, 2]] = D_svd
    V_svd = np.transpose(V_svd, [0, 2, 1])

    S = np.tile(np.eye(3), (N, 1, 1))
    mask = np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0
    S[mask, 2, 2] = -1

    R = np.matmul(U_svd, np.matmul(S, np.transpose(V_svd, [0, 2, 1])))
    s = 1.0/sigma2*np.trace(np.matmul(D_diag, S), axis1=1, axis2=2)
    t = mu_M.squeeze(1) - (s[:, None, None] * np.matmul(R, np.transpose(mu_D, axes=[0, 2, 1]))).squeeze(2)

    return s, R, t
