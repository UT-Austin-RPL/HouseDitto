import os, glob, shutil
from pathlib import Path
import numpy as np
import random
random.seed(42)
rng = np.random.default_rng(42)


joint_types = ["prismatic", "revolute"]
data_train_path = "../dataset/cubicasa5k_objects/train/*.npz" 
data_test_path = "../dataset/cubicasa5k_objects/test/*.npz" 
output_path = "../dataset/cubicasa5k_objects_processed/%s"


# create dir
for joint_type in joint_types:
    assert len(glob.glob(os.path.join(output_path % joint_type, 'train', 'scenes', '*'))) == 0
    assert len(glob.glob(os.path.join(output_path % joint_type, 'test', 'scenes', '*'))) == 0
    if not os.path.exists(output_path % joint_type):
        os.makedirs(os.path.join(output_path % joint_type, 'train', 'scenes'))
        os.makedirs(os.path.join(output_path % joint_type, 'test', 'scenes'))


# get fname_list
fname_list_train = {k: [] for k in joint_types}
for fname in glob.glob(data_train_path):
    npy = np.load(fname, allow_pickle=True)
    if npy["joint_type"] == 0:
        fname_list_train["revolute"].append(fname)
    elif npy["joint_type"] == 1:
        fname_list_train["prismatic"].append(fname)
    else:
        raise not NotImplementedError

for joint_type in joint_types:
    fname_list_train[joint_type].sort()
    fname_list_train[joint_type] = np.array(fname_list_train[joint_type])
    

# get fname_list
fname_list_test = {k: [] for k in joint_types}
for fname in glob.glob(data_test_path):
    npy = np.load(fname, allow_pickle=True)
    if npy["joint_type"] == 0:
        fname_list_test["revolute"].append(fname)
    elif npy["joint_type"] == 1:
        fname_list_test["prismatic"].append(fname)
    else:
        raise not NotImplementedError

for joint_type in joint_types:
    fname_list_test[joint_type].sort()
    fname_list_test[joint_type] = np.array(fname_list_test[joint_type])


for joint_type in joint_types:
    with open(os.path.join(output_path % joint_type, 'train.txt'), 'w') as f:
        f.writelines('\n'.join(fname_list_train[joint_type]))
        
    with open(os.path.join(output_path % joint_type, 'test.txt'), 'w') as f:
        f.writelines('\n'.join(fname_list_test[joint_type]))

print("List of joint types:", joint_types)
print("Train (revolute/prismatic): %d/%d" % (len(fname_list_train['revolute']), len(fname_list_train['prismatic'])))
print("Test (revolute/prismatic): %d/%d" % (len(fname_list_test['revolute']), len(fname_list_test['prismatic'])))

for joint_type in joint_types:
    for fname in fname_list_train[joint_type]:
        shutil.copy2(fname, os.path.join(output_path % joint_type, 'train', 'scenes'))
    for fname in fname_list_test[joint_type]:
        shutil.copy2(fname, os.path.join(output_path % joint_type, 'test', 'scenes'))