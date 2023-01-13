import os
import numpy as np
import imageio
import torch 
import glob

from imresize import imresize

original_folder = '/workspace/swinir_train/testsets/Set5/original'
LR_folder = '/workspace/swinir_train/testsets/Set5/LRbicx4_from_torch_version_fo_matlab'
dataname_list = [os.path.basename(data_path) for data_path in glob.glob(original_folder + '/*.png') ]


for dataname in dataname_list:
    image = imageio.imread(os.path.join(original_folder, dataname))
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    scale = 4

    if image.shape[1] % 4 != 0 or image.shape[2] % scale != 0:
        print("does not match size... ")
        print("problem image: ", dataname)

    image = imresize(image, sizes=(image.shape[1]//scale, image.shape[2]//scale))
    image = image.permute(1, 2, 0)
    image = image.numpy()
    imageio.imwrite(f"{LR_folder}/{dataname}", image)

print("complete!")