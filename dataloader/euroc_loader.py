import os
import sys
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
svo_lib_path = os.path.join(os.path.dirname(parent), 'svo-lib/build/svo_env')
sys.path.append(svo_lib_path)
import svo_env
import os
import csv
import glob
import yaml
import numpy as np
from scipy.spatial.transform import Rotation


test_split = {
        "MH_01_easy":      [1000, 3500],
        "MH_02_easy":      [500,  3000],
        "MH_03_medium":    [400,  2645],
        "MH_04_difficult": [340,  1900],
        "MH_05_difficult": [350,  2245],
        "V1_01_easy":      [0,    2860],
        "V1_02_medium":    [130,  1685],
        "V1_03_difficult": [0,    2050],
        "V2_01_easy":      [0,    2190],
        "V2_02_medium":    [100,  2290],
        "V2_03_difficult": [0,    1875],
}   # This needs to be alphebetically sorted



class EurocLoader:
    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None):
        assert mode == 'val'

        self.root_path = root_path
        self.num_envs = num_envs
        self.val_traj_ids = val_traj_ids
        self.test_split = list(test_split.keys())

        self.img_h, self.img_w = 480, 752

        if traj_name:
            print("[EuRoC Dataloader] Loading trajectory: %s" % traj_name)
            self.extract_trajectory(traj_name)
        else:
            print("[EuRoC Dataloader] Loading trajectories")
            print(test_split)
            self.extract_trajectories()

        self.extract_poses_images()

        self.set_up_running_indices()

    def is_test_scene(self, scene):
        return any(x in scene for x in self.test_split)

    def extract_trajectories(self):
        trajectories = glob.glob(os.path.join(self.root_path, '*'))
        self.trajectories_paths = [traj for traj in trajectories if self.is_test_scene(traj)]
        self.trajectories_paths.sort()

        if self.val_traj_ids != -1 and self.val_traj_ids is not None:
            self.trajectories_paths = [self.trajectories_paths[i] for i in self.val_traj_ids]

        assert self.num_envs == len(self.trajectories_paths)
    
    def extract_trajectory(self, traj_name):
        assert traj_name in test_split, "The requested trajectory (%s) is not available!" % traj_name
        self.trajectories_paths = [os.path.join(self.root_path, traj_name)]

    def matching_time_indices(self, stamps_1: np.ndarray, stamps_2: np.ndarray,
                              max_diff: float = 0.01,
                              offset_2: float = 0.0):
        """
        Adapted from: https://github.com/MichaelGrupp/evo/blob/master/evo/core/sync.py
        Searches for the best matching timestamps of two lists of timestamps
        and returns the list indices of the best matches.
        :param stamps_1: first vector of timestamps (numpy array)
        :param stamps_2: second vector of timestamps (numpy array)
        :param max_diff: max. allowed absolute time difference
        :param offset_2: optional time offset to be applied to stamps_2
        :return: 2 lists of the matching timestamp indices (stamps_1, stamps_2)
        """
        matching_indices_1 = []
        matching_indices_2 = []
        stamps_2 = stamps_2.copy()
        stamps_2 += offset_2
        for index_1, stamp_1 in enumerate(stamps_1):
            diffs = np.abs(stamps_2 - stamp_1)
            index_2 = int(np.argmin(diffs))
            if diffs[index_2] <= max_diff:
                matching_indices_1.append(index_1)
                matching_indices_2.append(index_2)
        return matching_indices_1, matching_indices_2

    def extract_poses_images(self):
        self.poses = {}
        self.image_filenames = {}
        self.img_timestamps_sec = {}
        for traj in self.trajectories_paths:
            with open(os.path.join(traj, 'mav0', 'cam0', 'data.csv'), "r") as file:
                image_data = list(csv.reader(file, delimiter=","))
            image_timestamps = np.array([int(row[0]) for row in image_data[1:]])
            seq_image_filenames =  [row[1] for row in image_data[1:]]

            # Crop trajectories to SVO evaluation ranges
            start_idx, end_idx = test_split[traj.split(os.sep)[-1]]
            image_timestamps = image_timestamps[start_idx:end_idx]
            seq_image_filenames = seq_image_filenames[start_idx:end_idx]

            # Synchronize images and poses
            gt_data = np.genfromtxt(os.path.join(traj, 'mav0', 'state_groundtruth_estimate0', 'data.csv'), delimiter=",")[:, :8]
            matching_indices_1, matching_indices_2 = self.matching_time_indices(image_timestamps, gt_data[:, 0], max_diff=100000)
            assert (matching_indices_1[-1] - matching_indices_1[0] + 1) == len(matching_indices_1)
            gt_data = gt_data[matching_indices_2, :]
            seq_image_filenames = [seq_image_filenames[i] for i in matching_indices_1]

            self.image_filenames['_'.join(traj.split(os.sep)[2:])] = seq_image_filenames
            assert gt_data.shape[0] == len(self.image_filenames['_'.join(traj.split(os.sep)[2:])])

            # Transform pose the SVO coordinate frame
            with open(os.path.join(traj, 'mav0', 'cam0', 'sensor.yaml'), 'r') as stream:
                sensor_data = yaml.safe_load(stream)
            T_B_cam = np.asarray(sensor_data['T_BS']['data']).reshape([4, 4])
            T_w_B = np.zeros([gt_data.shape[0], 4, 4])
            T_w_B[:, 3, 3] = 1
            T_w_B[:, :3, 3] = gt_data[:, 1:4]
            T_w_B[:, :3, :3] =  Rotation.from_quat(gt_data[:, [5, 6, 7, 4]]).as_matrix()

            T_w_cam_matrix = np.matmul(T_w_B, T_B_cam)
            T_w_cam = np.zeros([gt_data.shape[0], 7])
            T_w_cam[:, :3] = T_w_cam_matrix[:, :3, 3]
            T_w_cam[:, 3:] = Rotation.from_matrix(T_w_cam_matrix[:, :3, :3]).as_quat()
            self.poses['_'.join(traj.split(os.sep)[2:])] = T_w_cam

            self.img_timestamps_sec['_'.join(traj.split(os.sep)[2:])] = image_timestamps[matching_indices_1] * 1e-9

    def set_up_running_indices(self):
        self.traj_idx = np.arange(self.num_envs, dtype="int")

        self.start_dataloader_idx = np.zeros([self.num_envs])
        self.select_new_traj = np.ones([self.num_envs], dtype='bool')
        self.nr_samples_per_traj = np.zeros([len(self.trajectories_paths)])
        self.internal_idx = 0
        self.first_call = True

        for i_traj, (k,v) in enumerate(self.image_filenames.items()):
            self.nr_samples_per_traj[i_traj] = len(v)
        self.nr_samples = int(self.nr_samples_per_traj.sum())

    # get the timestamps of the images which have a ground truth pose
    def get_image_timestamps_in_sec(self, traj_name):
        assert ('EuRoC_' + traj_name) in self.img_timestamps_sec.keys(), "%s is not loaded!" % traj_name
        return self.img_timestamps_sec['EuRoC_' + traj_name]

    def __getitem__(self, idx):
        if self.first_call:
            self.first_call = False
        else:
            self.internal_idx += 1

        traj_time_idx = self.internal_idx - self.start_dataloader_idx

        # No new trajectory is selected if the old one is finished. Each env processes the same trajectory
        self.select_new_traj = np.logical_or(traj_time_idx >= self.nr_samples_per_traj[self.traj_idx],
                                             self.select_new_traj)
        self.start_dataloader_idx[self.select_new_traj] = self.internal_idx
        new_sequence_mask = self.select_new_traj.copy()
        self.select_new_traj[:] = False

        traj_time_idx = (self.internal_idx - self.start_dataloader_idx)
        traj_time_idx = traj_time_idx.astype('int')

        poses = np.zeros([self.num_envs, 7])
        image_paths = []
        for i in range(self.num_envs):
            traj_name = '_'.join(self.trajectories_paths[self.traj_idx[i]].split(os.sep)[2:])
            poses[i, :] = self.poses[traj_name][traj_time_idx[i], :]
            image_paths.append(os.path.join(self.trajectories_paths[self.traj_idx[i]], 'mav0', 'cam0', 'data',
                                            self.image_filenames[traj_name][traj_time_idx[i]]))

        images = svo_env.load_image_batch(image_paths, self.num_envs, self.img_h, self.img_w)

        return images, poses, new_sequence_mask

    def __len__(self):
        return int(self.nr_samples_per_traj.max())
