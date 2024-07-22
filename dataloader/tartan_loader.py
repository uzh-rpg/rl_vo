import os
import sys
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
svo_lib_path = os.path.join(os.path.dirname(parent), 'svo-lib/build/svo_env')
sys.path.append(svo_lib_path)
import svo_env
import os
import glob
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Same as Deep Patch VO
test_split = [
    "abandonedfactory/Easy/P011",
    "abandonedfactory/Hard/P011",
    "abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/Hard/P014",
    "amusement/Easy/P008",
    "amusement/Hard/P007",
    "carwelding/Easy/P007",
    "endofworld/Easy/P009",
    "gascola/Easy/P008",
    "gascola/Hard/P009",
    "hospital/Easy/P036",
    "hospital/Hard/P049",
    "japanesealley/Easy/P007",
    "japanesealley/Hard/P005",
    "neighborhood/Easy/P021",
    "neighborhood/Hard/P017",
    "ocean/Easy/P013",
    "ocean/Hard/P009",
    "office2/Easy/P011",
    "office2/Hard/P010",
    "office/Hard/P007",
    "oldtown/Easy/P007",
    "oldtown/Hard/P008",
    "seasidetown/Easy/P009",
    "seasonsforest/Easy/P011",
    "seasonsforest/Hard/P006",
    "seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/Hard/P018",
    "soulcity/Easy/P012",
    "soulcity/Hard/P009",
    "westerndesert/Easy/P013",
    "westerndesert/Hard/P007",
]


class TartanLoader:
    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None):
        self.mode = mode
        self.root_path = root_path
        self.num_envs = num_envs
        self.val_traj_ids = val_traj_ids
        self.test_split = test_split
        self.R_tartan_svo_quat = Rotation.from_matrix(np.array([[0, 0, 1],
                                                                [1, 0, 0],
                                                                [0, 1, 0]]))

        self.img_h, self.img_w = 480, 640

        if traj_name:
            print("[TartanAir Dataloader] Loading trajectory: %s" % traj_name)
            self.extract_trajectory(traj_name)
        else:
            self.extract_trajectories()

        self.extract_poses()
        self.set_up_running_indices()


    def is_test_scene(self, scene):
        return any(x in scene for x in self.test_split)

    def extract_trajectories(self):
        trajectories = glob.glob(os.path.join(self.root_path, '*/*/*'))
        trajectories = [traj for traj in trajectories if 'calibration' not in traj]
        if self.mode == 'train':
            self.trajectories_paths = [traj for traj in trajectories if not self.is_test_scene(traj)]
        elif self.mode == 'val':
            self.trajectories_paths = [traj for traj in trajectories if self.is_test_scene(traj)]
            self.trajectories_paths.sort()

            if self.val_traj_ids != -1 and self.val_traj_ids is not None:
                self.trajectories_paths = [self.trajectories_paths[i] for i in self.val_traj_ids]

            assert self.num_envs == len(self.trajectories_paths)
        else:
            raise ValueError('Mode {} not defined'.format(self.mode))
        
    def extract_trajectory(self, traj_name):
        self.trajectories_paths = [os.path.join(self.root_path, traj_name)]

    def extract_poses(self):
        self.poses = {}
        self.img_timestamps_sec = {}
        for traj in self.trajectories_paths:
            # Transform pose the SVO coordinate frame
            # TartanAir: the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right,
            # the z-axis is pointing to the camera's downward.
            T_w_t = np.genfromtxt(os.path.join(traj, 'pose_left.txt'))
            R_w_t_quat = Rotation.from_quat(T_w_t[:, 3:])
            R_w_svo_quat = (R_w_t_quat * self.R_tartan_svo_quat).as_quat()
            T_w_t[:, 3:] = R_w_svo_quat

            self.poses['_'.join(traj.split(os.sep)[2:])] = T_w_t

    def set_up_running_indices(self):
        self.traj_idx = np.arange(self.num_envs, dtype="int")
        self.start_dataloader_idx = np.zeros([self.num_envs])
        self.select_new_traj = np.ones([self.num_envs], dtype='bool')
        self.nr_samples_per_traj = np.zeros([len(self.trajectories_paths)])
        self.internal_idx = 0

        for i_traj, traj_path in enumerate(self.trajectories_paths):
            self.nr_samples_per_traj[i_traj] = len(os.listdir(os.path.join(traj_path, 'image_left_gray')))
        self.nr_samples = int(self.nr_samples_per_traj.sum())

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        return np.array(image)[:, :, ::-1]
    
    # get the timestamps of the images which have a ground truth pose
    def get_image_timestamps_in_sec(self, traj_name):
        traj = '_'.join(traj_name.split(os.sep))
        assert ('TartanAir_' + traj) in self.img_timestamps_sec.keys(), "%s is not loaded!" % traj_name
        return self.img_timestamps_sec['TartanAir_' + traj]

    def __getitem__(self, idx):
        self.internal_idx += 1
        traj_time_idx = self.internal_idx - self.start_dataloader_idx

        self.select_new_traj = np.logical_or(traj_time_idx >= self.nr_samples_per_traj[self.traj_idx],
                                             self.select_new_traj)
        # Select new trajectories
        if self.mode == 'train':
            self.traj_idx[self.select_new_traj] = torch.randint(0, len(self.trajectories_paths),
                                                                [self.select_new_traj.sum()]).numpy()

        self.start_dataloader_idx[self.select_new_traj] = self.internal_idx
        new_sequence_mask = self.select_new_traj.copy()
        self.select_new_traj[:] = False

        traj_time_idx = (self.internal_idx - self.start_dataloader_idx)
        traj_time_idx = traj_time_idx.astype('int')

        image_paths = [os.path.join(self.trajectories_paths[self.traj_idx[i]], 'image_left_gray',
                                      '{:06d}_left.jpg'.format(traj_time_idx[i])) for i in range(self.num_envs)]

        images = svo_env.load_image_batch(image_paths, self.num_envs, self.img_h, self.img_w)

        poses = np.zeros([self.num_envs, 2, 7])
        poses_idx = np.stack([traj_time_idx,
                              np.minimum(traj_time_idx + 1, self.nr_samples_per_traj[self.traj_idx] - 1)], axis=1)
        poses_idx = poses_idx.astype('int')
        for i in range(self.num_envs):
            poses[i, :, :] = self.poses['_'.join(self.trajectories_paths[self.traj_idx[i]].split(os.sep)[2:])][poses_idx[i], :]

        return images, poses, new_sequence_mask

    def __len__(self):
        return int(self.nr_samples / self.num_envs)
