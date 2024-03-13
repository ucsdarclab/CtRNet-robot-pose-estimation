#!/usr/bin/env python3
import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia
import shutil

import torch
import torchvision.transforms as transforms
from multiprocessing.pool import Pool
from itertools import repeat

from PIL import Image as PILImage
from utils import *
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import transforms3d as t3d
import tf2_ros
from sklearn.metrics.pairwise import rbf_kernel
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

import matplotlib.patches as mpatches
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'

################################################################
import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros/"
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 7
args.scale = 0.15625
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

if args.use_gpu:
    device = "cuda"
else:
    device = "cpu"

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CtRNet = CtRNet(args)

def preprocess_img(cv_img,args):
    image_pil = PILImage.fromarray(cv_img)
    width, height = image_pil.size
    new_size = (int(width*args.scale),int(height*args.scale))
    image_pil = image_pil.resize(new_size)
    image = trans_to_tensor(image_pil)
    return image

shutil.rmtree("/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals")
os.mkdir("/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals")
#############################################################################3

#start = time.time()
point_samples = []
rotation_samples = []
prev_confidence = []
cTr_minus_one = None
visual_idx = 0
def gotData(img_msg, joint_msg):
    #global start
    global point_samples
    global rotation_samples
    global prev_confidence
    global cTr_minus_one
    global points_2d_minus_one
    
    # print("Received data!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        if args.use_gpu:
            image = image.cuda()

        cTr, points_2d, segmentation, confidence = CtRNet.inference_single_image(image, joint_angles)
        # print(cTr)
        # update_publisher(cTr, img_msg)
        qua = kornia.geometry.conversions.angle_axis_to_quaternion(cTr[:,:3]).detach().cpu() # xyzw
        T = cTr[:,3:].detach().cpu()
        # filtering_method = "none"
        filtering_method = "particle"
        joint_confident_thresh = 3
        num_joint_confident = torch.sum(torch.gt(confidence, 0.90))
        if filtering_method == "none":
            # input("Press Enter to continue")
            update_publisher(cTr, img_msg, qua.numpy().squeeze(), T.numpy().squeeze())
            return
        elif filtering_method == "particle":
            if cTr_minus_one is not None:
                pred_cTr = particle_filter(points_2d, cTr_minus_one, cTr, joint_angles, 0.02, 3000, 3, image)
                cTr_minus_one = pred_cTr
                points_2d_minus_one = points_2d
                pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(pred_cTr[:,:3]).detach().cpu() # xyzw
                pred_T = pred_cTr[:,3:].detach().cpu()
                update_publisher(cTr, img_msg, pred_qua.numpy().squeeze(), pred_T.numpy().squeeze())
            else:
                print(f"Skipping because t-1 is {cTr_minus_one}")
                cTr_minus_one = cTr
                points_2d_minus_one = points_2d
            return
        # avg_confidence = torch.mean(confidence)
        # print(avg_confidence)
        if num_joint_confident < joint_confident_thresh:
            print(f"Only confident with {num_joint_confident} joints, skipping...")
            return

        if len(point_samples) < 30:
            print(f"Number of samples: {len(point_samples)}")
            point_samples.append(T)
            rotation_samples.append(qua)
            prev_confidence = confidence
            return
        
        # confidence_diff = prev_confidence - confidence
        # print(confidence_diff)

        is_not_outlier = None
        thresh = 1.5
        if filtering_method == "z_score":
            is_not_outlier, max_pos_z, max_rot_z = z_score(T, qua, thresh)
        elif filtering_method == "mod_z_score":
            is_not_outlier, max_pos_z, max_rot_z = mod_z_score(T, qua, thresh)
        else:
            print("Invalid filtering method")

        keep_outlier_prob = torch.min(torch.tensor(1), torch.sqrt(torch.max(max_pos_z, max_rot_z) - 1) / 3)
        keep_outlier = random.random() < keep_outlier_prob
        if is_not_outlier:
            print("Kept cTr")
            point_samples.pop(0)
            rotation_samples.pop(0)
            point_samples.append(T)
            rotation_samples.append(qua)
            update_publisher(cTr, img_msg, qua.numpy().squeeze(), T.numpy().squeeze())
        elif keep_outlier:
            print("Skipped but kept outlier")
            # print(keep_outlier_prob)
            point_samples.pop(0)
            rotation_samples.pop(0)
            point_samples.append(T)
            rotation_samples.append(qua)
        else:
            print("Skipped cTr")

        #### visualization code ####
        #points_2d = points_2d.detach().cpu().numpy()
        #img_np = to_numpy_img(image)
        #img_np = overwrite_image(img_np,points_2d[0].astype(int), color=(1,0,0))
        #plt.imsave("test.png",img_np)
        ####


    except CvBridgeError as e:
        print(e)
def visualize_panda(particles, joint_angles, cTr, image, points_2d, max_w_idx, points_2d_minus_one):
    global visual_idx
    base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros"
    mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
              base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
             ]
    robot_renderer = CtRNet.setup_robot_renderer(mesh_files)
    robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
    rendered_image = CtRNet.render_single_robot_mask(cTr.squeeze().detach().cuda(), robot_mesh, robot_renderer)
    img_np = to_numpy_img(image)
    img_np = 0.0* np.ones(img_np.shape) + img_np * 0.6
    red = (1,0,0)
    green = (0,1,0)
    blue = (0,0,1)
    yellow = (1,1,0)
    img_np = overwrite_image(img_np, particles.reshape(-1, particles.shape[-1]), color=blue, point_size=1)
    img_np = overwrite_image(img_np, particles[max_w_idx, :, :], color=red, point_size=1)
    img_np = overwrite_image(img_np, points_2d.detach().cpu().numpy().squeeze().astype(int), color=green, point_size=3)
    img_np = overwrite_image(img_np, points_2d_minus_one.detach().cpu().numpy().squeeze().astype(int), color=yellow, point_size=3)

    plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1)
    plt.title("keypoints")
    plt.imshow(img_np)
    # plt.subplot(1,3,2)
    # plt.title("segmentation")
    # plt.imshow(segmentation.squeeze().detach().cpu().numpy())
    # plt.subplot(1,3,2)
    # plt.title("rendering")
    # plt.imshow(rendered_image.squeeze().detach().cpu().numpy())
    colors = [blue, red, green, yellow]
    labels = ["Projected particles", "Max particle", "Current point2d", "Previous point2d"]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    plt.savefig(f"/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals/result{visual_idx}.png", dpi=800, format="png")
    visual_idx += 1
    input("Type Enter to continue")

def particle_filter(points_2d, cTr_minus_one, cTr, joint_angles, sigma, m, steps, image):
    # print("--------------------------------------")
    print("Particle filter")
    particles_r = None
    particles_t = None
    pred_cTr = torch.zeros((1, 6))
    for i in range(steps):
        # print(f"resample step {i}")
        normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([sigma]))
        omega_r = normal.sample((m, 3, 3)).squeeze()
        omega_t = normal.sample((m, 3)).squeeze()

        # Step 1
        if particles_r is None:
            rvec_minus_one = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr_minus_one[:, :3]).detach().cpu()
            tvec_minus_one = cTr_minus_one[:,3:].detach().cpu()

            # m x 3 tensor of particles
            particles_r = rvec_minus_one.repeat(m,1,1)
            particles_t = tvec_minus_one.repeat(m,1)

        particles_r += omega_r
        particles_t += omega_t

        # Step 2
        _, t_list = CtRNet.robot.get_joint_RT(joint_angles)
        points_3d = torch.from_numpy(np.array(t_list)).float()
        p_t = points_3d[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
        K = torch.from_numpy(CtRNet.intrinsics)
        z_t_hats = []

        # p = Pool(processes=3)
        # args = zip(p_t.repeat(m,1,1).numpy(), np.float64(particles_r.cpu().detach().numpy()), np.float64(particles_t.cpu().detach().numpy()), K.repeat(m,1,1).numpy(), repeat(None))
        # results = p.starmap(cv2.projectPoints, args)
        # p.close()
        # p.join()
        # print(results)
         #z_t_hats = [z_t_hat for (z_t_hat, _) in results]
        # z_t_hats = list(list(zip(*results))[0]) 
        for j in range(m): 
            z_t_hat, _ = cv2.projectPoints(p_t.numpy(), np.float64(particles_r[j, :, :].cpu().detach().numpy()), np.float64(particles_t[j, :].cpu().detach().numpy()), K.numpy(), None)
            z_t_hats.append(z_t_hat)

        # Step 3
        z_t = points_2d.reshape(1, 14)
        z_t_hats_points = np.array(z_t_hats).squeeze(2)
        z_t_hats = z_t_hats_points.reshape(m, -1)
        w = rbf_kernel(z_t_hats, Y=z_t.cpu().detach().numpy()) #avoid zeros
        
        # resampling
        # w_norm = w.squeeze() / np.sum(w.squeeze())
        # resample_idxs = systematic_resample(w_norm)
        # resample_idxs = stratified_resample(w_norm)
        # particles_r[:] = particles_r[resample_idxs]
        # particles_t[:] = particles_t[resample_idxs]

        # if i == steps-1:
            # weighted normalized ctr
            # cvTr = torch.eye(4).repeat(m, 1, 1)
            # cvTr[:, :3, :3] = particles_r
            # cvTr[:, :3, 3] = particles_t
            # w_cvTr = w[:, :, None] * cvTr.detach().numpy()
            # sum_w_cvTr = torch.sum(torch.from_numpy(w_cvTr), 0)
            # sum_w = torch.sum(torch.from_numpy(w))
            # norm_sum_cvTr = sum_w_cvTr / sum_w
            # rvec = norm_sum_cvTr[:3, :3].contiguous()
            # tvec = norm_sum_cvTr[:3, 3]
            # pred_cTr = torch.zeros((1,6))
            # pred_cTr[0, :3] = kornia.geometry.conversions.rotation_matrix_to_angle_axis(rvec)
            # pred_cTr[0, 3:] = tvec

            # max weighted ctr
        max_w_idx = np.argmax(w)
        pred_cTr = torch.zeros((1, 6))
            # print(particles_r[max_w_idx, : ,:])
        pred_cTr[0, :3] = kornia.geometry.conversions.rotation_matrix_to_angle_axis(particles_r[max_w_idx, :, :])
        pred_cTr[0, 3:] = particles_t[max_w_idx, :]
            
    visualize_panda(z_t_hats_points, joint_angles, cTr_minus_one, image, points_2d, max_w_idx, points_2d_minus_one)

    return pred_cTr.detach()

def z_score(T, qua, thresh):
    point_mean = torch.mean(torch.stack(point_samples), dim=0)
    point_std = torch.std(torch.stack(point_samples), dim=0)
    rotation_mean = torch.mean(torch.stack(rotation_samples), dim=0)
    rotation_std = torch.std(torch.stack(rotation_samples), dim=0)
    point_z_score  = (T - point_mean) / point_std
    rotation_z_score  = (qua - rotation_mean) / rotation_std

    test_keep_stable_point = not torch.any(torch.gt(torch.abs(point_z_score), thresh))
    test_keep_stable_rotation = not torch.any(torch.gt(torch.abs(rotation_z_score), thresh))
    if not (test_keep_stable_point and test_keep_stable_rotation):
        print(f"test points z {point_z_score}") 
        print(f"test rotations z {rotation_z_score}") 
    return (test_keep_stable_point and test_keep_stable_rotation), torch.max(torch.abs(point_z_score)), torch.max(torch.abs(rotation_z_score))

def median_abs_deviation(data):
    data_stack = torch.stack(data, dim=0)
    data_mean = torch.mean(data_stack)
    data_mean_repeated = data_mean.unsqueeze(0).repeat(data_stack.shape[0],1)
    mad = torch.median(torch.abs(data_stack - data_mean_repeated))
    return mad

def mod_z_score(T, qua, thresh):
    point_mean = torch.mean(torch.stack(point_samples), dim=0)
    point_mad = median_abs_deviation(point_samples)
    rotation_mean = torch.mean(torch.stack(rotation_samples), dim=0)
    rotation_mad = median_abs_deviation(rotation_samples)

    point_mod_z_score  = (0.6745*(T - point_mean)) / point_mad
    rotation_mod_z_score  = (0.6745*(qua - rotation_mean)) / rotation_mad

    test_keep_stable_point = not torch.any(torch.gt(torch.abs(point_mod_z_score), thresh))
    test_keep_stable_rotation = not torch.any(torch.gt(torch.abs(rotation_mod_z_score), thresh))
    if not (test_keep_stable_point and test_keep_stable_rotation):
        print(f"test points z {point_mod_z_score}") 
        print(f"test rotations z {rotation_mod_z_score}") 
    return (test_keep_stable_point and test_keep_stable_rotation), torch.max(torch.abs(point_mod_z_score)), torch.max(torch.abs(rotation_mod_z_score))

def update_publisher(cTr, img_msg, qua, T):
    p = geometry_msgs.msg.PoseStamped()
    p.header = img_msg.header
    p.pose.position.x = T[0]
    p.pose.position.y = T[1]
    p.pose.position.z = T[2]
    p.pose.orientation.x = qua[0]
    p.pose.orientation.y = qua[1]
    p.pose.orientation.z = qua[2]
    p.pose.orientation.w = qua[3]
    #print(p)
    pose_pub.publish(p)

    # TODO: Not using the filtered output!
    # Rotating to ROS format
    cvTr= np.eye(4)
    cvTr[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
    cvTr[:3, 3] = np.array(cTr[:, 3:].detach().cpu())

    # ROS camera to CV camera transform
    cTcv = np.array([[0, 0 , 1, 0], [-1, 0, 0 , 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T = cTcv@cvTr
    qua = t3d.quaternions.mat2quat(T[:3, :3]) # wxyz
    # Publish Transform
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera_base"
    t.child_frame_id = "panda_link0"
    t.transform.translation.x = T[0, 3]
    t.transform.translation.y = T[1, 3]
    t.transform.translation.z = T[2, 3]
    t.transform.rotation.x = qua[1]
    t.transform.rotation.y = qua[2]
    t.transform.rotation.z = qua[3]
    t.transform.rotation.w = qua[0]
    br.sendTransform(t)

rospy.init_node('panda_pose')
# Define your image topic
image_topic = "/rgb/image_raw"
robot_joint_topic = "/joint_states"
robot_pose_topic = "robot_pose"
# Set up your subscriber and define its callback
#rospy.Subscriber(image_topic, sensor_msgs.msg.Image, gotData)

image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
ats.registerCallback(gotData)


# Main loop:
rate = rospy.Rate(30) # 30hz

while not rospy.is_shutdown():
    rate.sleep()