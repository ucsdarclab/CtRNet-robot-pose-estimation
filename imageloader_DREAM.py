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
import json

from utils import find_ndds_data_in_dir, transform_DREAM_to_CPLSim_TCR


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_camera_parameters(data_folder):
    _, ndds_data_configs = find_ndds_data_in_dir(data_folder)
    with open(ndds_data_configs['camera'], "r") as json_file:
        data = json.load(json_file)

    fx = data['camera_settings'][0]['intrinsic_settings']['fx']
    fy = data['camera_settings'][0]['intrinsic_settings']['fy']
    cx = data['camera_settings'][0]['intrinsic_settings']['cx']
    cy = data['camera_settings'][0]['intrinsic_settings']['cy']

    return fx, fy, cx, cy


class ImageDataLoaderSynthetic(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)

        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            TCR_ndds = np.array(data['objects'][0]['pose_transform'])
            base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)

            joint_angle = torch.tensor(joint_angle, dtype=torch.float)
            base_to_cam = torch.tensor(base_to_cam, dtype=torch.float)

        else:
            joint_angle = None
            base_to_cam = None


        return image, joint_angle, base_to_cam




class ImageDataLoaderReal(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)

        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)


        else:
            joint_angle = None


        return image, joint_angle

    def get_data_with_keypoints(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)


        else:
            joint_angle = None

        keypoints = data['objects'][0]['keypoints']

        return image, joint_angle, keypoints

