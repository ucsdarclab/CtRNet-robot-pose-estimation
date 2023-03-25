import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from utils import *

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.data_folder = '/home/jingpei/Desktop/robot_pose_estimation/data_generation/baxter_data'
    args.base_dir = "/home/jingpei/Desktop/CtRNet-robot-pose-estimation"
    args.use_gpu = True
    args.trained_on_multi_gpus = False
    args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/pretrain/baxter/net.pth")
    #args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/baxter/net.pth")
    args.urdf_file = os.path.join(args.base_dir,"urdfs/Baxter/baxter_description/urdf/baxter.urdf")

    ##### training parameters #####
    args.batch_size = 6
    args.num_workers = 6
    args.lr = 1e-6
    args.beta1 = 0.9
    args.n_epoch = 500
    args.out_dir = 'outputs/Baxter_arm/weights'
    args.ckp_per_epoch = 10
    args.reproj_err_scale = 1.0 / 100.0
    ################################

    args.robot_name = 'Baxter_left_arm' # "Panda" or "Baxter_left_arm"
    args.n_kp = 7
    args.scale = 0.3125
    args.height = 1536
    args.width = 2048
    args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args

def main(args):
    ######## setup CtRNet ########
    from models.CtRNet import CtRNet

    CtRNet = CtRNet(args)

    mesh_files = [os.path.join(args.base_dir,"urdfs/Baxter/S0/S0.obj"), 
                os.path.join(args.base_dir,"urdfs/Baxter/S1/S1.obj"), 
                os.path.join(args.base_dir,"urdfs/Baxter/E0/E0.obj"), 
                os.path.join(args.base_dir,"urdfs/Baxter/E1/E1.obj"), 
                os.path.join(args.base_dir,"urdfs/Baxter/W0/W0.obj"), 
                os.path.join(args.base_dir,"urdfs/Baxter/W1/W1.obj"),
                os.path.join(args.base_dir,"urdfs/Baxter/W2/W2.obj")]

    robot_renderer = CtRNet.setup_robot_renderer(mesh_files)


    ######## setup dataset ########
    from imageloaders.baxter import ImageDataLoaderReal

    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train','eval']:
        datasets[phase] = ImageDataLoaderReal(data_folder = args.data_folder, scale = args.scale, trans_to_tensor = trans_to_tensor)


        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=args.batch_size,
            shuffle=True if phase == 'train' else False,
            num_workers=args.num_workers)

        data_n_batches[phase] = len(dataloaders[phase])

    ######## setup optimizer and criterions ########

    criterionMSE_sum = torch.nn.MSELoss(reduction='sum')
    criterionMSE_mean = torch.nn.MSELoss(reduction='mean')
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    criterions = {"mse_sum": criterionMSE_sum, "mse_mean": criterionMSE_mean, "bce": criterionBCE}

    optimizer = optim.Adam(CtRNet.keypoint_seg_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    ######## training loop ########
    epoch_writer = SummaryWriter(comment="_writter")

    best_valid_loss = np.inf

    for epoch in range(0, args.n_epoch):
        phases = ['train','eval']

        for phase in phases:
            iter_writer = SummaryWriter(comment="_epoch_" + str(epoch) + "_" + phase)

            # set model to train/eval mode
            CtRNet.keypoint_seg_predictor.train(phase == 'train')
            print("model training: " + str(CtRNet.keypoint_seg_predictor.training))


            meter_loss = AverageMeter()

            loader = dataloaders[phase]

            for i, data in tqdm(enumerate(loader), total=data_n_batches[phase]):

                if args.use_gpu:
                    if isinstance(data, list):
                        data = [d.cuda() for d in data]
                    else:
                        data = data.cuda()

                # load data
                img, joint_angles = data

                # forward
                loss = CtRNet.train_on_batch(img, joint_angles.cpu().squeeze(), robot_renderer, criterions, phase)
                meter_loss.update(loss.item(), n=img.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(CtRNet.keypoint_seg_predictor.parameters(), 10)
                    optimizer.step()

                # write to log
                iter_writer.add_scalar('loss_all', loss.item(), i)

            log = '%s [%d/%d] Loss: %.6f, LR: %f' % (
                phase, epoch, args.n_epoch,
                meter_loss.avg,
                get_lr(optimizer))

            iter_writer.close()

            print(log)
            if phase == 'train':
                epoch_writer.add_scalar('loss_train', meter_loss.avg, epoch)
            else:
                epoch_writer.add_scalar('loss_eval', meter_loss.avg, epoch)


            if phase == 'eval':
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg

                    torch.save(CtRNet.keypoint_seg_predictor.state_dict(), '%s/net_best.pth' % (args.out_dir))

                log = 'Best eval: %.6f' % (best_valid_loss)
                print(log)
                torch.save(CtRNet.keypoint_seg_predictor.state_dict(), '%s/net_last.pth' % (args.out_dir))
                
    epoch_writer.close()


if __name__ == '__main__':
    args = get_args()
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)