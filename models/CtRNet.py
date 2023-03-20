import torch
import torch.nn.functional as F
import kornia
import numpy as np

from .keypoint_seg_resnet import KeyPointSegNet
from .BPnP import BPnP

class CtRNet(torch.nn.Module):
    def __init__(self, args):
        super(CtRNet, self).__init__()

        self.args = args

        if args.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # load keypoint segmentation model
        self.keypoint_seg_predictor = KeyPointSegNet(args, use_gpu=args.use_gpu)

        if args.use_gpu:
            self.keypoint_seg_predictor = self.keypoint_seg_predictor.cuda()

        if args.robot_name == "Panda":
            self.keypoint_seg_predictor = torch.nn.DataParallel(self.keypoint_seg_predictor, device_ids=[0])
        
        if args.keypoint_seg_model_path is not None:
            print("Loading keypoint segmentation model from {}".format(args.keypoint_seg_model_path))
            self.keypoint_seg_predictor.load_state_dict(torch.load(args.keypoint_seg_model_path))

        if args.evaluate == True:
            self.keypoint_seg_predictor.eval()

        # load BPnP
        self.bpnp = BPnP.apply

        # set up camera intrinsics

        self.intrinsics = np.array([[   args.fx,    0.     ,    args.px   ],
                                    [   0.     ,    args.fy,    args.py   ],
                                    [   0.     ,    0.     ,    1.        ]])
        
        self.K = torch.tensor(self.intrinsics, device=self.device, dtype=torch.float)


        # set up robot model
        if args.robot_name == "Panda":
            from .robot_arm import PandaArm
            self.robot = PandaArm(args.urdf_file)
        elif args.robot_name == "Baxter_left_arm":
            from .robot_arm import BaxterLeftArm
            self.robot = BaxterLeftArm(args.urdf_file)


    def inference_single_image(self, img, joint_angles):
        # img: (1, 3, H, W)
        # joint_angles: (7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img)
        foreground_mask = torch.sigmoid(segmentation)
        _,t_list = self.robot.get_joint_RT(joint_angles)
        points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
        if self.args.robot_name == "Panda":
            points_3d = points_3d[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are dummy links

        #init_pose = torch.tensor([[  1.5497,  0.5420, -0.3909, -0.4698, -0.0211,  1.3243]])
        #cTb = bpnp(points_2d_pred, points_3d, K, init_pose)
        cTb = self.bpnp(points_2d, points_3d, self.K)

        return cTb, points_2d, foreground_mask
    
    def cTr_to_pose_matrix(self, cTr):
        # cTr: (1, 6)
        # pose_matrix: (4, 4)
        pose_matrix = torch.eye(4, device=self.device)
        pose_matrix[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3])
        pose_matrix[:3, 3] = cTr[:, 3:]
        return pose_matrix



