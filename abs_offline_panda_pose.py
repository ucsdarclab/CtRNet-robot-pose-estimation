#!/usr/bin/env python3
import sys
import os
import warnings

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia
import itertools
import torch
import torchvision.transforms as transforms

from PIL import Image as PILImage
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import transforms3d as t3d
import tf2_ros

from ros_nodes.abs_particle_filter import *
from ros_nodes.abs_probability_funcs import *
import imageio
from tqdm import tqdm
import json
import shutil
from scipy.spatial.transform import Rotation as R
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
################################################################
import argparse
base_dir = os.path.abspath(".")
sys.path.append(base_dir)
parser = argparse.ArgumentParser()
parser.add_argument("--filename")
# parser.add_argument("--orbslam", action="store_true")
# parser.add_argument("--dpvo", action="store_true")
parser.add_argument("--vo", type=str)
parser.add_argument("--nopf", action="store_true")
parser.add_argument("--justvo", action="store_true")
parser.add_argument("--novo", action="store_true")
parser.add_argument("--saveframes", action="store_true")
parser.add_argument("--num_p", type=int, default=1000)
parser.add_argument("--ep", type=str)
args = parser.parse_args()
# args.checkpoint = "/home/co-tracker/checkpoints/cotracker3.pth"
args.base_dir = "/home/workspace/src/orig-ctrnet"
args.confidence_threshold = 0.15
args.data_folder = ""

args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda'
args.n_kp = 7
args.scale = 0.3125
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 967.2597045898438, 967.2623291015625, 1024.62451171875, 772.18994140625

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
    return image, image_pil

LIGHT_BLUE_PRED = np.array([20, 139, 173], dtype=np.uint8)
LIGHT_BLUE_GT   = np.array([135, 20, 173], dtype=np.uint8)
CONTRAST_FACTOR = 10

def overlay_mask_on_frame(
    original_frame,           # HxWx3 uint8 (RGB)
    rendered_mask_rgb,        # HxWx3 uint8 (RGB), same size or will be resized
    gt_extr,
    alpha=0.5,
    blur_kernel_size=41,
    sigma=5,
    _scratch=None             # internal: dict of reusable buffers
):
    """
    Faster version:
    - keeps uint8 until blend
    - thresholds once and reuses binary mask
    - avoids repeated allocations
    """
    if _scratch is None:
        _scratch = {}

    h, w = original_frame.shape[:2]
    if rendered_mask_rgb.shape[:2] != (h, w):
        rendered_mask_rgb = cv2.resize(rendered_mask_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # Gray then binary once
    mask_gray = cv2.cvtColor(rendered_mask_rgb, cv2.COLOR_RGB2GRAY)  # uint8
    _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    light_blue = LIGHT_BLUE_GT if gt_extr else LIGHT_BLUE_PRED

    if '_color_full' not in _scratch or _scratch['_color_full'].shape[:2] != (h, w):
        _scratch['_color_full'] = np.empty_like(original_frame)
    _color_full = _scratch['_color_full']
    _color_full[:] = light_blue

    blue_mask = cv2.bitwise_and(_color_full, _color_full, mask=mask_bin)  # uint8

    blue_mask_f = blue_mask.astype(np.float32)
    cv2.subtract(blue_mask_f, 128.0, blue_mask_f)
    blue_mask_f *= CONTRAST_FACTOR
    cv2.add(blue_mask_f, 128.0, blue_mask_f)
    np.clip(blue_mask_f, 0, 255, out=blue_mask_f)
    blue_mask = blue_mask_f.astype(np.uint8)

    blurred = cv2.GaussianBlur(mask_bin, (blur_kernel_size, blur_kernel_size), sigma)  # uint8
    blurred_f = (blurred.astype(np.float32) / 255.0)  # 0..1 single channel

    orig_f = original_frame.astype(np.float32)

    overlay = cv2.addWeighted(orig_f, 1.0 - alpha, blue_mask.astype(np.float32), alpha, 0.0)

    # final = (1 - blurred)*orig + blurred*overlay
    # Broadcast single-channel blurred_f to 3 channels via multiplication
    one_minus = 1.0 - blurred_f
    final_f = orig_f * one_minus[..., None] + overlay * blurred_f[..., None]
    final_frame = final_f.astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour_color = (105, 25, 175) if gt_extr else (50, 100, 210)
        contour_thickness = 8
        # Comment out next 2 lines for more speed if not needed:
        # contour_smoothing_factor = 0.001
        # contours = [cv2.approxPolyDP(c, contour_smoothing_factor * cv2.arcLength(c, True), True) for c in contours]
        cv2.drawContours(final_frame, contours, -1, contour_color, contour_thickness)

    return final_frame


def overwrite_image(image, points_predicted, color=(0, 255, 0), point_size=8):
    # If many points, consider drawing small filled squares with cv2.polylines or stamping a precomputed disk
    pts = points_predicted.astype(np.int32).squeeze()
    for p in pts:
        cv2.circle(image, tuple(p), point_size, color, -1)
    return image
import numpy as np
import cv2

def draw_points_sized_by_weight(
    image_bgr,
    pts_xy,                 # shape [N,2], float or int pixel coords
    weights,                # shape [N], nonnegative
    color=(255, 200, 0),    # BGR
    r_min=2,                # smallest radius in pixels
    r_max=8,                # largest radius in pixels
    gamma=1.0,              # >1 emphasizes big weights, <1 spreads small ones
    quantile_clip=(0.05, 0.95),  # clip weights to robust range before scaling
    antialias=True,
):
    """
    Draws points with radius proportional to weight. In-place on image_bgr.
    Complexity ~ O(N * r^2). No per-pixel alpha blending (fast).
    """
    if len(pts_xy) == 0:
        return image_bgr

    # -- normalize weights to [0,1] robustly --
    w = np.asarray(weights, dtype=np.float32)
    w = np.maximum(w, 0.0)

    # avoid one huge weight making others tiny
    lo = np.quantile(w, quantile_clip[0]) if w.size > 4 else w.min()
    hi = np.quantile(w, quantile_clip[1]) if w.size > 4 else w.max()
    if hi <= lo + 1e-12:
        w01 = np.ones_like(w) * 0.5  # all the same if no spread
    else:
        w01 = np.clip((w - lo) / (hi - lo), 0.0, 1.0)

    # contrast shaping
    if gamma != 1.0:
        w01 = np.power(w01, gamma)

    # map to integer radii
    radii = (r_min + (r_max - r_min) * w01).astype(np.int32)
    radii = np.clip(radii, r_min, r_max)

    # -- draw (fast: just filled circles) --
    pts = np.rint(np.asarray(pts_xy)).astype(np.int32)
    H, W = image_bgr.shape[:2]
    lt = cv2.LINE_AA if antialias else cv2.LINE_8

    # small ROI bound check to skip off-image draws
    for (x, y), r in zip(pts, radii):
        if r <= 0: 
            continue
        if (x + r) < 0 or (y + r) < 0 or (x - r) >= W or (y - r) >= H:
            continue
        cv2.circle(image_bgr, (int(x), int(y)), int(r), color, thickness=-1, lineType=lt)

    return image_bgr

def visualize_panda(all_images, all_joint_angles, all_cTr, all_proj_points, all_points_2d, all_ct_points_2d, all_weights, filename):
    # Prefer OpenCV VideoWriter for speed (but keep imageio if you need consistent codec across envs)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # vw = cv2.VideoWriter(f'./visualization/{filename}.mp4', fourcc, 60, (args.width, args.height))
    writer = imageio.get_writer(f"./visualization/{filename}.mp4", fps=30)
    # writer_lossless = imageio.get_writer(f"./visualization/{filename}_lossless.mp4", fps=30)
    if args.saveframes:
        frames_dir = f"./visualization/{filename}_frames/"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.mkdir(frames_dir)

    mesh_files = [
        base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
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

    # Collect masks in a list; cat once
    mask_list = []

    # Reusable scratch buffers for overlay to avoid reallocations
    _scratch = {}

    # Precompute flags & size once
    gt_extr = not (args.justvo or args.novo or args.nopf)
    target_size = (args.width, args.height)

    for i in tqdm(range(len(all_images)), desc="saving visualization"):
        # Render robot mask (keep on device; move to CPU once per frame)
        rendered = CtRNet.render_single_robot_mask(
            all_cTr[i].squeeze(),
            robot_renderer.get_robot_mesh(all_joint_angles[i]),
            robot_renderer
        )

        # Save mask tensor for later (keep as original dtype/device for speed)
        mask_list.append(rendered)  # don't detach/cpu yet

        # Convert to uint8 RGB for overlay (do one detach->cpu only when needed)
        final_image = rendered.squeeze().detach().cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)           # assume in [0,1]
        # If rendered is already single-channel mask, skip color convert; just promote to 3ch later
        # Here you converted with RGB2BGR previously; if your pipeline expects RGB, skip this swap:
        # final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)  # (likely unnecessary)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)

        # Resize inputs once
        # lossless_img = all_images[i]
        if all_images[i].shape[1] != target_size[0] or all_images[i].shape[0] != target_size[1]:
            resized_img = cv2.resize(all_images[i], target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = all_images[i]

        # lossless_image_rgb = final_image_rgb
        if final_image_rgb.shape[1] != target_size[0] or final_image_rgb.shape[0] != target_size[1]:
            final_image_rgb = cv2.resize(final_image_rgb, target_size, interpolation=cv2.INTER_LINEAR)

        # Draw previous frame's points onto current image (as in your code)
        if all_proj_points and i > 0 and all_proj_points[i - 1] is not None:
            red = (255, 0, 0)
            # resized_img = overwrite_image(resized_img, all_proj_points[i - 1].squeeze(), color=red, point_size=3)
            resized_img = draw_points_sized_by_weight(resized_img, all_proj_points[i - 1].squeeze(), all_weights[i-1], color=red, r_min=1, r_max=3)
            # upscaled_points_2d = all_points_2d[i - 1]/args.scale
            # lossless_img = overwrite_image(all_images[i], upscaled_points_2d, color=red, point_size=5)

        if all_points_2d and i > 0 and all_points_2d[i - 1] is not None:
            green = (0, 255, 0)
            blue = (0, 0, 255)
            resized_img = overwrite_image(resized_img, all_points_2d[i - 1], color=green, point_size=3)
            if all_ct_points_2d[i - 1] is not None:
                # upscaled_ct_points_2d = all_ct_points_2d[i - 1]/args.scale
                # lossless_img = overwrite_image(lossless_img, upscaled_ct_points_2d, color=blue, point_size=5)
                resized_img = overwrite_image(resized_img, all_ct_points_2d[i - 1], color=blue, point_size=3)
            # upscaled_points_2d = all_points_2d[i - 1]/args.scale
            # lossless_img = overwrite_image(all_images[i], upscaled_points_2d, color=green, point_size=5)
        overlay_frame = overlay_mask_on_frame(
            resized_img, final_image_rgb, gt_extr, alpha=0.5, _scratch=_scratch
        )
        # overlay_frame_lossless = overlay_mask_on_frame(lossless_img, lossless_image_rgb, gt_extr, alpha=0.5, _scratch=_scratch)
        if args.saveframes:
            # overlay_frame is RGB; convert to BGR only if using cv2.imwrite
            bgr_overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{frames_dir}/overlay_{str(i).zfill(4)}.png", bgr_overlay_frame)
            # bgr_overlay_frame_lossless = cv2.cvtColor(overlay_frame_lossless, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"{frames_dir}/overlay_lossless_{str(i).zfill(4)}.png", bgr_overlay_frame_lossless)
            bgr_image_frame = cv2.cvtColor(all_images[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{frames_dir}/raw_{str(i).zfill(4)}.png", bgr_image_frame)

        writer.append_data(overlay_frame)  # RGB uint8
        # writer_lossless.append_data(overlay_frame_lossless)
        # vw.write(bgr_overlay_frame)  # if using OpenCV writer

    writer.close()
    # writer_lossless.close()
    # vw.release()

    # Single cat at the end; move to cpu once
    all_masks = torch.cat([m.detach().cpu() for m in mask_list], dim=0)
    torch.save(all_masks, f"masks/our_masks/{args.filename}.pt")

def process_step_query(window_frames, is_first_step, query):
    # print(len(window_frames))
    # print(window_frames[0].shape)
    video_chunk = (
        torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=device)
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return model(
        video_chunk,
        is_first_step=is_first_step,
        queries=query[None],
    )
#############################################################################3
#start = time.time()
new_data = False
points_2d = None
joint_angles = None
cTr = None
cam_pose = None
skipped = False
classification_result = None
cam_pos = None
cam_ori = None
new_cam_data = False
curr_image = None
if __name__ == "__main__":
    # if args.checkpoint is not None:
    #     model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    # else:
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(device)
    episode = args.ep
    vo_type = args.vo
    dataset_dir = "/home/workspace/src/ctrnet-x-rt/moving_panda_dataset/"
    episode_dir = os.path.join(dataset_dir, episode)
    info_file = os.path.join(episode_dir, "info.json")
    if vo_type == "dpvo":
        cam_pose_file = f"/home/workspace/src/ctrnet-x-rt/{vo_type}_case25_cams/{episode}/images.txt"
    elif vo_type == "orbslam":
        cam_pose_file = f"/home/workspace/src/ctrnet-x-rt/{vo_type}_cams/{episode}/images.txt"
    elif vo_type == "colmap":
        cam_pose_file = f"/home/workspace/src/ctrnet-x-rt/{vo_type}/{episode}.txt"

    print(f"Publishing joint and image data from {info_file}")
    with open(info_file) as f:
        data = json.load(f)
    steps = data["steps"]
    timestamps = list(steps.keys())
    all_images = []
    image_paths = []
    all_joint_angles = []
    for timestamp in timestamps:
        img_file = steps[timestamp]["img_file"]
        img_path = os.path.join(episode_dir, "images", img_file)
        image_paths.append(img_path)
        all_images.append(cv2.imread(img_path))
        all_joint_angles.append(np.array(steps[timestamp]["joint_angles"]))
    init_std = np.array([
                1.0e-2, 1.0e-2, 1.0e-2, # ori
                1.0e-3, 1.0e-3, 1.0e-3, # pos
            ])
    pf = ParticleFilter(num_states=6,
                        init_distribution=sample_gaussian,
                        motion_model=additive_gaussian,
                        obs_model=point_feature_obs,
                        num_particles=args.num_p)
    # rospy.loginfo("Initialized particle filter")

    prev_cTr = None
    use_particle_filter = not args.nopf
    prev_cam_pose = None
    all_cTr = []
    all_proj_points = []
    points_2d_window = torch.empty(1, 0, 2).to(device)
    points_3d_window = torch.empty(0, 3).to(device)
    all_v_points_2d = []
    all_v_images = []
    all_v_joint_angles = []
    start_time = time.time()
    T_cam_total = None
    window_size = 80
    # CoTracker
    window_frames = []
    all_v_ct_points = []
    all_v_weights = []
    is_first_step = True
    cotracker_query = None
    cotracker_points = None
    cotracker_vis = None
    pred_tracks = None
    warnings.filterwarnings("ignore", message="`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead.")
    # Processing episode
    for i in tqdm(range(0, len(all_images))):
        cv_img = cv2.cvtColor(all_images[i], cv2.COLOR_BGR2RGB)
        image, image_pil = preprocess_img(cv_img,args)
        curr_image = cv_img
        classification_result = {"end-effector": True, "base": True}
        joint_angles = all_joint_angles[i]
        if not args.novo and not args.nopf:
            cam_pose = cam_poses[i]
        all_v_images.append(curr_image)
        all_v_joint_angles.append(joint_angles)
        if not args.justvo or prev_cTr is None:
            with torch.no_grad():
                cTr, points_2d, mask, confidence = CtRNet.inference_single_image(image, joint_angles)
                all_v_points_2d.append(points_2d.cpu().numpy())
                if i == 0:
                    pf.init_filter(init_std, cTr.cpu().numpy())

        if use_particle_filter == False:
            # print("PARTICLE FILTER TURNED OFF")
            pred_T = cTr[:,3:].detach().cpu()
            all_cTr.append(cTr)
            continue

        if prev_cTr is None:
            prev_cTr = cTr
            prev_cam_pose = cam_pose
            prev_cTr_R = kornia.geometry.conversions.angle_axis_to_rotation_matrix(prev_cTr[:, :3]).detach().cpu()
            prev_cTr_t = prev_cTr[:, 3:].detach().cpu()
            T_prev_cTr = np.eye(4,4)
            T_prev_cTr[:3, :3] = prev_cTr_R
            T_prev_cTr[:3, 3] = prev_cTr_t
            all_cTr.append(cTr)
            continue
        
        if args.justvo:
            pred_cTr = T_prev_cTr @ T_cam_total
            # pred_cTr = T_cam_total @ T_prev_cTr
            pred_cTr_aa = torch.zeros((1, 6)).to(device)
            pred_cTr_aa[0, :3] = kornia.geometry.conversions.rotation_matrix_to_angle_axis(torch.from_numpy(pred_cTr[:3, :3]).contiguous()).detach().cpu()
            pred_cTr_aa[0, 3:] = torch.from_numpy(pred_cTr[:3, 3])
            all_cTr.append(pred_cTr_aa)
            continue

        # CoTracker Start ##################################################################################
        window_frames.append(cv_img)
        # Start CoTracker on high confidence points
        if cotracker_query is None:
            joint_confident_thresh = 7
            num_joint_confident = torch.sum(torch.gt(confidence, 0.85))
            if num_joint_confident >= joint_confident_thresh:
                print("Created cotracker query")
                cotracker_query = torch.cat(((torch.ones((7,1))*i).to(device), (points_2d / args.scale).squeeze().to(device)), 1)
                print(len(window_frames))
                print(cotracker_query)
        if i % model.step == 0 and i != 0 and cotracker_query is not None:
            pred_tracks, pred_visibility = process_step_query(
                window_frames,
                is_first_step,
                query=cotracker_query
            )
            is_first_step = False
        if pred_tracks is not None:
            cotracker_points = pred_tracks[:, -1, :, :]
            cotracker_vis = pred_visibility[:, -1, :].cpu().numpy()
        cotracker_points_2d = (cotracker_points * args.scale).cpu().numpy() if cotracker_points is not None else None
        all_v_ct_points.append(cotracker_points_2d)
        # CoTracker End ##################################################################################        
        # Predict Particle filter
        pred_std = np.array([6.0e-3, 6.0e-3, 6.0e-3,
                            8.0e-3, 8.0e-3, 8.0e-3])
        # pred_std = np.array([1.0e-2, 1.0e-2, 1.0e-2,
        #                     3.0e-2, 3.0e-2, 3.0e-2])
        pf.predict(pred_std, None)
        # Update Particle filter
        cam = None
        gamma = 0.03
        proj_points, point_weights = pf.update(points_2d, cotracker_points_2d, cotracker_vis, CtRNet, joint_angles, cam, prev_cTr, gamma, classification_result, confidence)

        best_particle = pf.get_mean_particle()

        all_cTr.append(torch.from_numpy(best_particle).to(device))
        all_proj_points.append(proj_points)
        all_v_weights.append(point_weights)
    
    print(f"FPS: {len(all_images) / (time.time() - start_time)}")
    if args.filename is not None:
        visualize_panda(all_v_images, all_v_joint_angles, all_cTr, all_proj_points, all_v_points_2d, all_v_ct_points, all_v_weights, args.filename)