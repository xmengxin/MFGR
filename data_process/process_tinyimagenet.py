from os.path import join
import os
import shutil

dataset_dir = '/home/yiran/xxm/datasets/tiny-imagenet-200'
valdir = dataset_dir + '/val'
new_valdir = dataset_dir + '/val_folder'
if not os.path.exists(new_valdir):
    os.makedirs(new_valdir)

with open(join(valdir, 'val_annotations.txt'), 'r') as f:
    infos = f.read().strip().split('\n')
    infos = [info.strip().split('\t')[:2] for info in infos]

for img_name, folder_name in infos:
    img_new_path = join(new_valdir, folder_name, 'images')
    if not os.path.exists(img_new_path):
        os.makedirs(img_new_path)
    img_old_path = join(valdir, 'images', img_name)
    shutil.copy(img_old_path, img_new_path)
