import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import point_cloud_utils as pcu
from src.datamodules.components.pc_utils import PointcloudRotate, angle_axis


class AffordanceDataset(Dataset):
    def __init__(self, root, split, num_points=1024, transform=None):
        
        self.root = root
        with open(os.path.join(self.root, 'split/%s_scene.txt' % split)) as file:
            self.filenames = [line.rstrip() for line in file]
        self.num_points = num_points
        self.split = split
        
        # for train
        self.balanced_sampling = True
        self.transform = transforms.Compose([
            PointcloudRotate(axis=np.array([0, 0, 1]))
        ])
        
        # for val and test
        self.random_generator_val = np.random.default_rng(42)
        self.sizeof_voxel_pc = 0.03

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        npy = np.load(os.path.join(self.root, fname), allow_pickle=True)
        pos = npy['pts']
        x = npy['color']
        y = npy['affordance'].reshape(-1)
            
        # normalization
        xyz_min = npy['room_info'].item()['xyz_min']
        xyz_max = npy['room_info'].item()['xyz_max']
        xyz_max[2] = 2.4 # height
        xyz_min_max = np.vstack([xyz_min, xyz_max])
        xyz_min_max_pts_frame = np.stack([xyz_min_max[:, 1], xyz_min_max[:, 2], xyz_min_max[:, 0]], axis=-1)
        center = np.expand_dims(np.mean(xyz_min_max_pts_frame, axis = 0), 0).astype('float32')
        dist = np.max(np.sqrt(np.sum((xyz_min_max_pts_frame - center) ** 2, axis = 1)),0).astype('float32')
        pos = pos - center  # center
        pos = pos / dist    # scale
        
        # remove label -1
        if self.split == 'train' or self.split == 'val':  
            pos = pos[y!=-1]
            x = x[y!=-1]
            y = y[y!=-1]
        
        # random sampling
        selected_idx = np.arange(len(pos))
        if self.split == 'train':                        
            self.sizeof_voxel_pc = 0.01
            # voxel sampling
            pos, x, y, selected_idx = self.voxel_sampling(pos, x, y, selected_idx, sizeof_voxel_pc=self.sizeof_voxel_pc)
            # over-/down-sampling to fix the point number
            pos, x, y, selected_idx = self.over_down_sampling(pos, x, y, selected_idx)                        
        elif self.split == 'val':
            self.sizeof_voxel_pc = 0.01
            # voxel sampling
            pos, x, y, selected_idx = self.voxel_sampling(pos, x, y, selected_idx, sizeof_voxel_pc=self.sizeof_voxel_pc, random_generator=self.random_generator_val)
            # over-/down-sampling to fix the point number
            pos, x, y, selected_idx = self.over_down_sampling(pos, x, y, selected_idx, random_generator=self.random_generator_val)            
        elif self.split == 'test':            
            self.sizeof_voxel_pc = 0.01
            # voxel sampling
            pos, x, y, selected_idx = self.voxel_sampling(pos, x, y, selected_idx, sizeof_voxel_pc=self.sizeof_voxel_pc, random_generator=self.random_generator_val)
            # over-/down-sampling to fix the point number
            pos, x, y, selected_idx = self.over_down_sampling(pos, x, y, selected_idx, random_generator=self.random_generator_val)                        
        else:
            raise NotImplementedError
                
        # convert to tensor
        pos = torch.tensor(pos)
        x = torch.tensor(x)
        y = torch.tensor(y)
                
        return (pos, x, y, fname, selected_idx)
        
    def pos_neg_sampling(self, pos, x, y, pos_idx, neg_link_idx, neg_obj_idx, random_generator=None, pos_neg_rate=1.):
        
        # initialize generator
        if random_generator is None:
            rng = np.random.default_rng()
        else:
            rng = random_generator
        
        # pos/neg sampling        
        n_pos = len(pos_idx)
        n_neg = n_pos * pos_neg_rate
        n_neg_link = len(neg_link_idx)
        n_neg_obj = len(neg_obj_idx)
        
        neg_idx = []
        neg_idx.extend(neg_link_idx)
        
        if len(neg_idx) < n_neg and n_neg_obj > 0:
            n_sample = min(n_neg - len(neg_idx), n_neg_obj)
            selected_point_idxs = rng.choice(range(n_neg_obj), n_sample, replace=False)
            neg_idx.extend(neg_obj_idx[selected_point_idxs])
        
        all_idx = np.concatenate([pos_idx, neg_idx])
        
        pos = pos[all_idx]
        x = x[all_idx]
        y = y[all_idx]        
        
        return pos, x, y, all_idx
    
    def over_down_sampling(self, pos, x, y, idx, random_generator=None):
        
        # initialize generator
        if random_generator is None:
            rng = np.random.default_rng()
        else:
            rng = random_generator
                
        if len(pos) >= self.num_points:
            selected_point_idxs = rng.choice(range(len(pos)), self.num_points, replace=False)
            pos = pos[selected_point_idxs]
            x = x[selected_point_idxs]
            y = y[selected_point_idxs]
            idx = idx[selected_point_idxs]
        else:
            while(len(pos) < self.num_points):
                n_oversample = min(len(pos), self.num_points - len(pos))
                selected_point_idxs = rng.choice(range(len(pos)), n_oversample, replace=False)
                pos_oversample = pos[selected_point_idxs]
                x_oversample = x[selected_point_idxs]
                y_oversample = y[selected_point_idxs]
                idx_oversample = idx[selected_point_idxs]
                pos = np.concatenate([pos, pos_oversample])
                x = np.concatenate([x, x_oversample])
                y = np.concatenate([y, y_oversample])
                idx = np.concatenate([idx, idx_oversample])
        assert len(pos) == self.num_points
        
        return pos, x, y, idx
    
    def voxel_sampling(self, pos, x, y, idx, sizeof_voxel_pc, random_generator=None):
        
        # initialize generator
        if random_generator is None:
            rng = np.random.default_rng()
        else:
            rng = random_generator
            
        # downsample the pointcloud
        pos_sampled, _, _ = pcu.downsample_point_cloud_voxel_grid(sizeof_voxel_pc, points=pos)
        _, selected_point_idxs = pcu.k_nearest_neighbors(pos_sampled, pos, k=1)
        
        pos = pos[selected_point_idxs]
        x = x[selected_point_idxs]
        y = y[selected_point_idxs]
        idx = idx[selected_point_idxs]
        return pos, x, y, idx