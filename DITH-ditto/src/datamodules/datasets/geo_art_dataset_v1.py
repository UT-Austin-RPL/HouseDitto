import glob
import os
import random

import numpy as np
import torch
from numpy.lib.arraysetops import isin
from omegaconf import ListConfig
from torch.utils.data import Dataset

from src.utils.misc import occ_to_binary_label, sample_occ_points, sample_point_cloud
from src.utils.transform import Rotation


# different occ points and seg points
# represent articulation as dense joints
# include transformed surface points
# for testing, use all occ points and seg points
class GeoArtDatasetV1(Dataset):
    def __init__(self, opt):
        if isinstance(opt["data_path"], ListConfig):
            # multi class
            self.path_list = []
            for data_path in opt["data_path"]:
                self.path_list.extend(
                    glob.glob(
                        os.path.join(opt["data_dir"], data_path, "scenes", "*.npz")
                    )
                )
        else:
            self.path_list = glob.glob(
                os.path.join(opt["data_dir"], opt["data_path"], "scenes", "*.npz")
            )
        if opt.get("num_data"):
            random.shuffle(self.path_list)
            self.path_list = self.path_list[: opt["num_data"]]
        self.num_point = opt["num_point"]
        self.norm = opt.get("norm", False)
        
        self.start_end_flip = opt.get("start_end_flip", False)
        
        self.rand_rot = opt.get("rand_rot", False)
        if self.norm:
            self.norm_padding = opt.get("norm_padding", 0.1)

    def __getitem__(self, index):
        data = np.load(self.path_list[index], allow_pickle=True)
        
        pc_start, pc_start_idx = sample_point_cloud(data["pc_start"], self.num_point)
        pc_end, pc_end_idx = sample_point_cloud(data["pc_end"], self.num_point)
        color_start = data["color_start"][pc_start_idx]
        color_end = data["color_end"][pc_end_idx]
        pc_seg_label_start = data["pc_seg_start"][pc_start_idx]
        pc_seg_label_end = data["pc_seg_end"][pc_end_idx]
        state_start = data["state_start"]
        state_end = data["state_end"]
        screw_axis = data["screw_axis"]
        screw_moment = data["screw_moment"]
        joint_type = data["joint_type"]
        joint_index = data["joint_index"]
        move_direction = data["move_direction"]      
                
        # get affordance
        from pathlib import Path
        import point_cloud_utils as pcu
        from src.datamodules.datasets.utils import axis2transformation
        
        hit_pos = data["hit_pos"]
        hit_pos_start = np.array([hit_pos[1], hit_pos[2], hit_pos[0]])
        move_direction = data["move_direction"]
        move_direction = np.array([move_direction[1], move_direction[2], move_direction[0]])         
        
        hit_pos_end = hit_pos_start + move_direction * 0.25
        hit_pos_end = hit_pos_end.reshape(1, 3)
        
        def gaussian(dists, sigma=0.01, muu=0):
            dst = dists.reshape(-1, 1)
            dst = (dst-muu) / np.max(np.sqrt(sigma ** 2)) # normalize to [0, 1]            
            gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))
            return gauss
        
        radius_start = 0.05
        aff_start = np.zeros((len(pc_start), 1))
        dists = pcu.pairwise_distances(pc_start, hit_pos_start.reshape(1, 3))
        gauss = gaussian(dists[dists < radius_start], sigma=np.sqrt(radius_start))
        aff_start[dists < radius_start] = gauss
        
        radius_end = 0.1 #0.05
        aff_end = np.zeros((len(pc_end), 1))
        dists = pcu.pairwise_distances(pc_end, hit_pos_end.reshape(1, 3))
        gauss = gaussian(dists[dists < radius_end], sigma=np.sqrt(radius_end))
        aff_end[dists < radius_end] = gauss
        
        bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
        bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        scale = scale * (1 + self.norm_padding)

        pc_start = (pc_start - center) / scale
        pc_end = (pc_end - center) / scale

        # ((p - c) / s) X l
        # = (p X l) / s - (c X l) / s
        # = m / s - (c X l) / s
        screw_point = np.cross(screw_axis, screw_moment)
        screw_point = (screw_point - center) / scale

        screw_moment = np.cross(screw_point, screw_axis)

        # screw_moment = screw_moment / scale - np.cross(center, screw_axis) / scale
        if joint_type == 1:
            # prismatic joint, state change with scale
            state_start /= scale
            state_end /= scale

        # only change revolute joint
        # only change z-axis joints
        # if screw_axis[2] <= -0.9 and joint_type == 0:
        if screw_axis[1] <= -0.9 and joint_type == 0:
            screw_axis = -screw_axis
            screw_moment = -screw_moment
            state_start, state_end = state_end, state_start

        screw_point = np.cross(screw_axis, screw_moment)
        p2l_vec_start, p2l_dist_start = batch_perpendicular_line(
            pc_start, screw_axis, screw_point
        )
        p2l_vec_end, p2l_dist_end = batch_perpendicular_line(
            pc_end, screw_axis, screw_point
        )
        
        
        # add RGB channel
        pc_start = np.concatenate([pc_start, color_start, aff_start.reshape(-1, 1)], axis=1)
        pc_end = np.concatenate([pc_end, color_end, aff_end.reshape(-1, 1)], axis=1)
        

        return_dict = {
            "pc_start": pc_start,  # N, 3 or 6
            "pc_end": pc_end,
            "pc_start_color": color_start,
            "pc_end_color": color_end,
            "pc_seg_label_start": pc_seg_label_start,
            "pc_seg_label_end": pc_seg_label_end,
            "state_start": state_start,
            "state_end": state_end,
            "screw_axis": screw_axis,
            "screw_moment": screw_moment,
            "p2l_vec": p2l_vec_start,
            "p2l_dist": p2l_dist_start,
            "joint_type": joint_type,
            "joint_index": joint_index,
            "p_seg": pc_start,
            "seg_label": pc_seg_label_start,
            "scale": scale,
            "center": center,
            "data_path": self.path_list[index],
        }
                
        if self.start_end_flip:
            return_dict["pc_start"] = pc_end
            return_dict["pc_end"] = pc_start
            return_dict["pc_start_color"] = color_end
            return_dict["pc_end_color"] = color_start
            return_dict["pc_seg_label_start"] = pc_seg_label_end
            return_dict["pc_seg_label_end"] = pc_seg_label_start
            return_dict["state_start"] = state_end
            return_dict["state_end"] = state_start
            
            return_dict["p2l_vec"] = p2l_vec_end
            return_dict["p2l_dist"] = p2l_dist_end
            return_dict["p_seg"] = pc_end
            return_dict["seg_label"] = pc_seg_label_end
        for k, v in return_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = torch.from_numpy(v).float()
        return return_dict

    def __len__(self):
        return len(self.path_list)


def batch_perpendicular_line(
    x: np.ndarray, l: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """
    x: B * 3
    l: 3
    pivot: 3
    p_l: B * 3
    """
    offset = x - pivot
    p_l = offset.dot(l)[:, np.newaxis] * l[np.newaxis] - offset
    dist = np.sqrt(np.sum(p_l ** 2, axis=-1))
    p_l = p_l / (dist[:, np.newaxis] + 1.0e-5)
    return p_l, dist
