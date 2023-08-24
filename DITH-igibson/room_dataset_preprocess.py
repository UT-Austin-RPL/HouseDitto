import os
from tqdm import tqdm
import numpy as np
import point_cloud_utils as pcu

from utils.utils_misc import make_dirs


dataset_path = '../dataset/cubicasa5k_rooms/'
split_list = ['train', 'val', 'test']
save_path = '../dataset/cubicasa5k_rooms_processed/'
make_dirs(os.path.join(save_path, 'split'))
sizeof_voxel_pc = 0.01


for split in split_list:
    # get file list
    with open(os.path.join(save_path, 'split/%s_scene.txt' % split)) as file:
        fname_list = [line.rstrip() for line in file]

    # generation
    for fname in tqdm(fname_list):    
        npy = np.load(os.path.join(dataset_path, fname), allow_pickle=True)
        pos = npy['pts']
        x = npy['color']
        y = npy['affordance'].reshape(-1)
        
        # augment data by labeling neighbors as positive and the rest as negative
        if split == 'train' or split == 'val':  
            y_new = y
            for pt in pos[y==1]:
                dists = pcu.pairwise_distances(pos, pt.reshape(1, 3))
                y_new[dists < 0.005] = 1
            y = y_new
        
        new_npy = {k: npy[k] for k in dict(npy) if k in ['scene_name', 'room_name', 'room_info', 'pts', 'color', 'affordance', 'label']}
        new_npy['pts'] = pos
        new_npy['color'] = x
        new_npy['affordance'] = y
        np.savez_compressed(os.path.join(save_path, fname), **new_npy)