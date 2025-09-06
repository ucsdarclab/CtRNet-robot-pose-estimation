import numpy as np
import torch
import kornia
import cv2
from scipy import optimize
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel
use_gpu = True
if use_gpu:
    device = "cuda"
else:
    device = "cpu"
def to_transform(state):
    particle_ori = state[:, :3]
    particle_pos = state[:, 3:]

    particle_r = kornia.geometry.conversions.angle_axis_to_rotation_matrix(particle_ori.contiguous())
    particle_t = particle_pos.reshape(state.shape[0], 3, 1)

    last_row = torch.tile(torch.tensor([0,0,0,1]).to(device), (state.shape[0],1,1))
    particle_rt = torch.cat((particle_r, particle_t), axis=2)
    particle_transform = torch.cat((particle_rt, last_row), axis=1)
    return particle_transform.to(torch.float32)

def to_axis_angle(state):
    state_r = state[:, :3, :3]
    state_t = state[:, :3, 3]
    state_aa = kornia.geometry.conversions.rotation_matrix_to_angle_axis(state_r.contiguous())
    return torch.cat((state_aa, state_t), axis=1)

def sample_gaussian(state, std):
    m = torch.zeros(std.shape).to(device)
    sample = torch.normal(m, std)
    state_T = to_transform(state)
    sample_T = to_transform(sample)
    return to_axis_angle(sample_T @ state_T)

def additive_gaussian(state, std):
    new_state = sample_gaussian(state, std)
    return new_state

def projectPoints(points, K):
    # points is Nx7x3 np array
    num_particles = points.shape[0]
    K = torch.tile(K, (num_particles, 1, 1))
    points = points.permute((0,2,1))
    dehomog_pts = points / points[:,-1,:].unsqueeze(1)
    projected_point = torch.matmul(K, dehomog_pts).permute((0,2,1))[:,:,:-1]
    return projected_point
def project_points_intrinsics_torch(p_c: torch.Tensor, K) -> torch.Tensor:
    """
    p_c: [N,7,3] or [7,3] camera-frame 3D points (float tensor)
    K  : 3x3 intrinsics (torch or numpy)
    returns: [N,7,2] or [7,2] pixel coords (torch)
    """
    if not isinstance(K, torch.Tensor):
        K = torch.as_tensor(K)
    p = p_c.to(dtype=torch.float32)
    K = K.to(device=p.device, dtype=p.dtype)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = p[..., 0]; Y = p[..., 1]; Z = p[..., 2].clamp_min(1e-6)
    U = fx * (X / Z) + cx
    V = fy * (Y / Z) + cy
    return torch.stack((U, V), dim=-1)
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
    transformed_state = torch.concatenate((new_state_aa, new_state_t), axis=1)
    
    sample = sample_gaussian(std)
    new_state = transformed_state + sample
    # old_state = state + sample
    # new_state[sample_idx] = old_state[sample_idx]
    return new_state


def point_feature_obs(states, points_2d, ct_points_2d, ct_vis, ctrnet, joint_angles, cam, cTr, gamma, class_result, conf):
    num_particles = states.shape[0]
    _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    t_list = torch.from_numpy(t_list).to(torch.float32)
    p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    p_c_1 = to_transform(states)
    p_q_T = torch.transpose(torch.cat((p_t, torch.ones((p_t.shape[0], 1))), axis=1), 0, 1) # [7, 4]
    p_c_2 = torch.tile(p_q_T, (num_particles, 1, 1)).to(device)
    p_c = p_c_1 @ p_c_2
    p_c = p_c.permute(0,2,1)[:,:,:-1]
    projected_points = projectPoints(p_c, ctrnet.K.to(torch.float32))

    conf_thresh = 0.65
    vis_thresh=0.7
    sigma0=6.0
    eps=1e-6
    mask_pts = (conf >= conf_thresh)

    # per-point sigma_i for CtRNet
    sigma_ctr = sigma0 / torch.sqrt(torch.clip(conf, eps, 1.0))
    scale_ctr = torch.repeat_interleave(sigma_ctr, 2)
    mask_dim_ctr = torch.repeat_interleave(mask_pts, 2)

    proj_flat = projected_points.reshape(num_particles, -1)
    det_ctr   = points_2d.reshape(1, -1)
    diff_ctr  = (proj_flat - det_ctr) / torch.clamp(scale_ctr, min=1e-6)
    r2_ctr   = (diff_ctr[:, mask_dim_ctr]**2).sum(axis=1)
    M_ctr    = max(int(mask_pts.sum()) * 2, 1)
    r2_ctr  /= M_ctr
    sigma0_cotr=5.0
    if ct_points_2d is not None:
        det_cot = ct_points_2d.reshape(1, -1)
        sigma_cot = torch.full((7,), sigma0_cotr)
        scale_cot = torch.repeat_interleave(sigma_cot, 2).to(device)
        diff_cot = (proj_flat - det_cot) / torch.clamp(scale_cot, min=1e-6)
        r2_cot    = (diff_cot**2).sum(axis=1)
        M_cot     = 14
        r2_cot   /= M_cot
    else:
        r2_cot = 0.0
    if ct_points_2d is not None and ct_vis is not None:
        det_cot = ct_points_2d.reshape(1, -1)
        det_cot = torch.repeat_interleave(det_cot, num_particles, axis=0)
        mask_cot_pts = (ct_vis >= vis_thresh)
        mask_cot_dim = torch.repeat_interleave(mask_cot_pts, 2)
        if mask_cot_pts.sum() >= 4:
            sigma_cot = sigma0_cotr / torch.sqrt(torch.clip(ct_vis, eps, 1.0))
            scale_cot = torch.repeat_interleave(sigma_cot, 2)
            diff_cot = (proj_flat - det_cot) / scale_cot
            r2_cot = (diff_cot[:, mask_cot_dim] ** 2).sum(axis=1) if mask_cot_dim.any() else 0.0
        else:
            r2_cot = 0.0
    else:
        r2_cot = 0.0
    E = r2_ctr + r2_cot

    prob = torch.exp(-gamma * E)

    proj_points = projected_points.reshape(num_particles * 7, 2)
    return prob, proj_points