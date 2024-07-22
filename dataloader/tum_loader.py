import os
import sys
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
svo_lib_path = os.path.join(os.path.dirname(parent), 'svo-lib/build/svo_env')
sys.path.append(svo_lib_path)
import svo_env
import os
import glob
import numpy as np


test_split = {
        "rgbd_dataset_freiburg1_360":    [100, 756],
        "rgbd_dataset_freiburg1_desk":   [0,  613],
        "rgbd_dataset_freiburg1_desk2":  [0,  640],
        "rgbd_dataset_freiburg1_floor":  [0,  1242],
        "rgbd_dataset_freiburg1_plant":  [0,  1141],
        "rgbd_dataset_freiburg1_room":   [0,  1362],
        "rgbd_dataset_freiburg1_rpy":    [10, 723],
        "rgbd_dataset_freiburg1_teddy":  [0,  1419],
        "rgbd_dataset_freiburg1_xyz":    [0,  798],
}   # This needs to be alphebetically sorted


class TumLoader:
    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None):
        assert mode == 'val'
        self.root_path = root_path
        self.num_envs = num_envs
        self.val_traj_ids = val_traj_ids
        self.test_split = list(test_split.keys())

        self.img_h, self.img_w = 480, 640
        if traj_name:
            print("[TUM Dataloader] Loading trajectory: %s" % traj_name)
            self.extract_trajectory(traj_name)
        else:
            print("[TUM Dataloader] Loading trajectories")
            print(test_split)
            self.extract_trajectories()

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
            image_timestamps = []
            seq_image_filenames = []
            with open(os.path.join(traj, 'rgb.txt')) as file:
                for i_line, line in enumerate(file):
                    if i_line <= 2:
                        continue
                    split_line = line.rstrip().split(' ')
                    image_timestamps.append(float(split_line[0]))
                    seq_image_filenames.append(split_line[1])

            image_timestamps = np.array(image_timestamps)

            # Crop trajectories to SVO evaluation ranges
            start_idx, end_idx = test_split[traj.split(os.sep)[-1]]
            image_timestamps = image_timestamps[start_idx:end_idx]
            seq_image_filenames = seq_image_filenames[start_idx:end_idx]

            gt_data = np.genfromtxt(os.path.join(traj, 'groundtruth.txt'), delimiter=" ")
            gt_data[:, 4:] = gt_data[:, 4:] / np.linalg.norm(gt_data[:, 4:], axis=1)[:, None]
            # Synchronize images and poses
            matching_indices_1, matching_indices_2 = self.matching_time_indices(image_timestamps, gt_data[:, 0], max_diff=0.1)

            if not traj.split(os.sep)[-1] == 'rgbd_dataset_freiburg1_floor':
                assert (matching_indices_1[-1] - matching_indices_1[0] + 1) == len(matching_indices_1)
                gt_data = gt_data[matching_indices_2, :]
                self.image_filenames['_'.join(traj.split(os.sep)[2:])] = [seq_image_filenames[i] for i in matching_indices_1]
            else:
                matching_indices_1_ext = np.arange(matching_indices_1[-1] - matching_indices_1[0] + 1) + matching_indices_1[0]
                self.image_filenames['_'.join(traj.split(os.sep)[2:])] = [seq_image_filenames[i] for i in matching_indices_1_ext]
                ext_gt_data = -np.ones([len(matching_indices_1_ext), 8])
                ext_gt_data[matching_indices_1] = gt_data[matching_indices_2, :]
                gt_data = ext_gt_data

            assert gt_data.shape[0] == len(self.image_filenames['_'.join(traj.split(os.sep)[2:])])

            self.poses['_'.join(traj.split(os.sep)[2:])] = gt_data[:, 1:]

            self.img_timestamps_sec['_'.join(traj.split(os.sep)[2:])] = image_timestamps[matching_indices_1]

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
        assert ('TUM-RGBD_' + traj_name) in self.img_timestamps_sec.keys(), "%s is not loaded!" % traj_name
        return self.img_timestamps_sec['TUM-RGBD_' + traj_name]

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
            image_paths.append(os.path.join(self.trajectories_paths[self.traj_idx[i]],
                                            self.image_filenames[traj_name][traj_time_idx[i]]))

        images = svo_env.load_image_batch(image_paths, self.num_envs, self.img_h, self.img_w)

        return images, poses, new_sequence_mask

    def __len__(self):
        return int(self.nr_samples_per_traj.max())
