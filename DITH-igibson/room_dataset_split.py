import os, glob, shutil
import numpy as np

from utils.utils_misc import make_dirs


rng = np.random.default_rng(42)


dataset_path = '../dataset/cubicasa5k_rooms/'
save_path = '../dataset/cubicasa5k_rooms_processed/'
make_dirs(os.path.join(save_path, 'split'))

fname_list = [o for o in os.listdir(dataset_path)
                if not os.path.isdir(os.path.join(dataset_path,o))]
fname_list = np.array(fname_list)

n_sample = len(fname_list)
split_rate = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2,
}
n_train = int(n_sample * split_rate['train'])
n_val = int(n_sample * split_rate['val'])
n_test = n_sample - n_train - n_val

# shuffle
rng.shuffle(fname_list)

# split
fname_list_train = fname_list[:n_train]
fname_list_val = fname_list[n_train:n_train+n_val]
fname_list_test = fname_list[n_train+n_val:]

# sort
fname_list_train.sort()
fname_list_val.sort()
fname_list_test.sort()

# print
print("Train: ", len(fname_list_train))
print("Val: ", len(fname_list_val))
print("Test: ", len(fname_list_test))

# save
with open(os.path.join(save_path, 'split', 'train_scene.txt'), 'w') as f:
    f.writelines('\n'.join(fname_list_train))
    
with open(os.path.join(save_path, 'split', 'val_scene.txt'), 'w') as f:
    f.writelines('\n'.join(fname_list_val))

with open(os.path.join(save_path, 'split', 'test_scene.txt'), 'w') as f:
    f.writelines('\n'.join(fname_list_test))