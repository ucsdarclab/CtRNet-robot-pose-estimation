import numpy as np
import torch
import kornia
import cv2
from scipy import optimize
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel

def to_transform(state):
    particle_ori = state[:, :3]
    particle_pos = state[:, 3:]

    particle_r = kornia.geometry.conversions.angle_axis_to_rotation_matrix(torch.from_numpy(particle_ori)).numpy()
    particle_t = particle_pos.reshape(state.shape[0], 3, 1)

    last_row = np.tile(np.array([0,0,0,1]), (state.shape[0],1,1))
    particle_rt = np.concatenate((particle_r, particle_t), axis=2)
    particle_transform = np.concatenate((particle_rt, last_row), axis=1)
    return particle_transform

def to_axis_angle(state):
    state_r = state[:, :3, :3]
    state_t = state[:, :3, 3]
    state_aa = kornia.geometry.conversions.rotation_matrix_to_angle_axis(torch.from_numpy(state_r).contiguous()).numpy()
    return np.concatenate((state_aa, state_t), axis=1)

def sample_gaussian(state, std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    state_T = to_transform(state)
    sample_T = to_transform(sample)
    return to_axis_angle(sample_T @ state_T)

def additive_gaussian(state, std):
    new_state = sample_gaussian(state, std)
    return new_state

def projectPoints(points, K):
    # points is Nx7x3 np array
    num_particles = points.shape[0]
    K = np.tile(K, (num_particles, 1, 1))
    points = points.transpose((0,2,1))
    dehomog_pts = points / np.expand_dims(points[:,-1,:], 1)
    projected_point = np.matmul(K, dehomog_pts).transpose((0,2,1))[:,:,:-1]
    return projected_point

def camera_transform_gaussian(state, std, cam_transform):
    # cam_transform[:, :3, 3] = cam_transform[:, :3, 3] * np.linspace(0.008, 0.07, num=state.shape[0])[:, None]
    # sample_idx = np.random.choice(state.shape[0], size=500, replace=False)
    particle_transform = to_transform(state)
    camT = cam_transform if cam_transform.ndim == 2 else cam_transform[0]
    new_state = camT[None, :, :] @ particle_transform
    # new_state = particle_transform @ cam_transform

    new_state_r = new_state[:, :3, :3]
    new_state_t = new_state[:, :3, 3]
    new_state_aa = kornia.geometry.conversions.rotation_matrix_to_angle_axis(torch.from_numpy(new_state_r).contiguous()).numpy()
    transformed_state = np.concatenate((new_state_aa, new_state_t), axis=1)
    
    sample = sample_gaussian(std)
    new_state = transformed_state + sample
    # old_state = state + sample
    # new_state[sample_idx] = old_state[sample_idx]
    return new_state


def point_feature_obs(states, points_2d, ct_points_2d, ct_vis, ctrnet, joint_angles, cam, cTr, gamma, class_result, conf):
    #convert state to angle axis
    num_particles = states.shape[0]
    _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    K = np.float64(ctrnet.intrinsics)
    # np.set_printoptions(suppress=True)
    p_c_1 = to_transform(states)
    p_c_2 = np.tile(np.transpose(np.concatenate((p_t, np.ones((p_t.shape[0], 1))), axis=1)), (num_particles, 1, 1))
    p_c = np.matmul(p_c_1, p_c_2)
    p_c = p_c.transpose((0,2,1))[:,:,:-1]
    projected_points = projectPoints(p_c, K)
    # num_points = 7
    # proj_points = projected_points.reshape(num_particles*num_points, 2)
    # projected_points = projected_points.reshape(num_particles, num_points*2)
    # detected_points =  points_2d.cpu().detach().numpy()
    # detected_points = np.reshape(np.tile(detected_points, (num_particles, 1, 1)), (num_particles, num_points*2))
    # prob = rbf_kernel(projected_points, Y=detected_points, gamma=0.01).squeeze()
    # Make association between detected and projected points to compute prob and use prob to update weights
        # Flatten to [N,14]
    proj_flat = projected_points.reshape(num_particles, -1)  # [N,14]
    det = points_2d.detach().cpu().numpy().astype(np.float32).reshape(1, -1)  # [1,14]
    det_flat = np.repeat(det, num_particles, axis=0)  # [N,14]

    # Confidence -> per-keypoint sigma_i; scale each (x,y) by sigma_i
    conf_thresh = 0.8
    vis_thresh=0.3
    sigma0=4.0
    eps=1e-6
    conf_np = conf.detach().cpu().numpy().astype(np.float32)  # [7]
    mask = conf_np >= conf_thresh
    mask_pts = (conf_np >= conf_thresh)                       # [7] which keypoints to use
    # if mask.sum() < 4:
    #     # too few reliable points -> very flat likelihood to avoid collapse
    #     return np.full((num_particles,), 1e-6, dtype=np.float32), projected_points.reshape(num_particles*7, 2)

    # sigma = sigma0 / np.sqrt(np.clip(conf_np, eps, 1.0))  # [7]
    # scale = np.repeat(sigma, 2)  # [14]

    # proj_n = proj_flat / scale  # per-dimension normalization
    # det_n  = det_flat  / scale
    # per-point σ_i for CtRNet
    sigma_ctr = sigma0 / np.sqrt(np.clip(conf_np, eps, 1.0))   # [7]
    scale_ctr = np.repeat(sigma_ctr, 2).astype(np.float32)     # [14]
    mask_dim_ctr = np.repeat(mask_pts, 2)                      # [14]

    proj_flat = projected_points.reshape(num_particles, -1)    # [N,14]
    det_ctr   = points_2d.detach().cpu().numpy().astype(np.float32).reshape(1, -1)  # [1,14]
    det_ctr   = np.repeat(det_ctr, num_particles, axis=0)      # [N,14]
    
    # whiten (divide) then squared error over used dims
    diff_ctr = (proj_flat - det_ctr) / np.maximum(scale_ctr, 1e-6)  # [N,14]
    r2_ctr   = (diff_ctr[:, mask_dim_ctr]**2).sum(axis=1)           # [N]
    M_ctr    = max(int(mask_pts.sum()) * 2, 1)                      # active dims
    r2_ctr  /= M_ctr                                                # dim-normalize
    # -- CoTracker measurement (optional, visibility-aware) --
    sigma0_cotr=3.0
    if ct_points_2d is not None:
        det_cot = ct_points_2d.reshape(1, -1).astype(np.float32)    # [1,14]
        det_cot = np.repeat(det_cot, num_particles, axis=0)         # [N,14]

        # If you have per-point visibility 'cotr_vis' in [0,1], use soft weighting:
        # lam = (cotr_vis**2).clip(0,1)                              # [7], optional
        # sigma_cot = (sigma0_cotr / np.sqrt(np.maximum(lam, eps))) + 12.0*(1.0 - lam)
        # Otherwise, a single σ for all CoTracker joints is fine:
        sigma_cot = np.full(7, sigma0_cotr, dtype=np.float32)       # [7]

        scale_cot = np.repeat(sigma_cot, 2).astype(np.float32)      # [14]
        diff_cot  = (proj_flat - det_cot) / np.maximum(scale_cot, 1e-6)
        r2_cot    = (diff_cot**2).sum(axis=1)                       # [N]
        M_cot     = 14                                              # two dims * 7 joints
        r2_cot   /= M_cot
    else:
        r2_cot = 0.0
        # -- CoTracker measurement (optional, visibility-aware) --
    if ct_points_2d is not None and ct_vis is not None:
        det_cot = ct_points_2d.reshape(1, -1).astype(np.float32)    # [1,14]
        det_cot = np.repeat(det_cot, num_particles, axis=0)         # [N,14]
        vis_np = ct_vis.astype(np.float32)                        # [7]
        mask_cot_pts = (vis_np >= vis_thresh)
        mask_cot_dim = np.repeat(mask_cot_pts, 2)                                          # [14]
        if mask_cot_pts.sum() >= 4:
            sigma_cot = sigma0_cotr / np.sqrt(np.clip(vis_np, eps, 1.0))                   # [7]
            scale_cot = np.repeat(sigma_cot, 2).astype(np.float32)                         # [14]
            diff_cot = (proj_flat - det_cot) / scale_cot                                   # [N,14]
            r2_cot = (diff_cot[:, mask_cot_dim] ** 2).sum(axis=1) if mask_cot_dim.any() else 0.0
        else:
            r2_cot = 0.0
    else:
        r2_cot = 0.0
    # --- Combine purely via σ (no w_*) ---
    # "Energy" = sum of squared whitened residuals from both sources
    E = r2_ctr + r2_cot                                            # [N]

    # Tempering (optional, helps on “jerk” frames)
    beta = 1   # try 0.6–0.9 when large inter-frame motion is detected

    prob = np.exp(-beta * gamma * E).astype(np.float32)            # [N]

    proj_points = projected_points.reshape(num_particles * 7, 2)
    return prob, proj_points