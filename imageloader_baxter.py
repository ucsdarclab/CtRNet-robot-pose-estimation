import os
import time

from PIL import Image

import cv2
import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import glob
import pickle


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




class ImageDataLoaderReal(Dataset):

    def __init__(self, data_folder, scale, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.joint_angles = np.load(os.path.join(self.data_folder,"gt_joint.npy"))

        self.image_files = glob.glob(os.path.join(self.data_folder,"images","*.png"))

        self.scale = scale



    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder,"images",str(idx) + ".png")
        image_pil = pil_loader(img_path)
        width, height = image_pil.size
        new_size = (int(width*self.scale),int(height*self.scale))
        image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        joint_angle = torch.tensor(self.joint_angles[idx], dtype=torch.float)



        return image, joint_angle
    

class ImageDataLoaderSynthetic(Dataset):

    def __init__(self, data_file, data_dir, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_file = data_file
        self.data_dir = data_dir



        infile = open(data_file,'rb')
        self.data_dict = pickle.load(infile)
        infile.close()

        self.image_files = list(self.data_dict.keys())



    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_dir, self.data_dict[self.image_files[idx]]['image'])
        image_pil = pil_loader(img_path)
        image = self.trans_to_tensor(image_pil)

        joint_angle = torch.tensor(self.data_dict[self.image_files[idx]]['joint_angle'], dtype=torch.float)
        base_to_cam = torch.tensor(self.data_dict[self.image_files[idx]]['base_to_cam'], dtype=torch.float)

        return image, joint_angle, base_to_cam