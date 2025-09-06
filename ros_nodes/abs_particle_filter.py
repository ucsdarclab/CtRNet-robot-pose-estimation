#!/usr/bin/env python3
import torch
from filterpy import monte_carlo
use_gpu = True
if use_gpu:
    device = "cuda"
else:
    device = "cpu"
class ParticleFilter:
    def __init__(self, num_states, init_distribution, motion_model, obs_model, num_particles, min_num_effective_particles=2):
        self._particles = torch.zeros((num_particles, num_states)).to(device)
        self._weights = torch.ones((num_particles), dtype=torch.float32).to(device)/torch.tensor(num_particles, dtype=torch.float32).to(device)
        self._min_num_effective_particles = min_num_effective_particles

        self._init_distribution = init_distribution
        self._motion_model = motion_model
        self._obs_model = obs_model

        self._prev_joint_angles = None
        self._since_resample = 0
        self.resample_hist = []
        self._num_particles = num_particles

    def norm_weights(self):
        self._weights = self._weights/torch.sum(self._weights)

    def init_filter(self, std, cTr):
        tiled_std = torch.tile(std, (self._num_particles, 1))
        tiled_cTr = torch.tile(cTr, (self._num_particles, 1))
        self._particles = self._init_distribution(tiled_cTr, tiled_std)

    def predict(self, std, cam_transform=None):
        tiled_std = torch.tile(std, (self._num_particles, 1))
        if cam_transform is None:
            self._particles = self._motion_model(self._particles, tiled_std)
        else:
            tilded_cam_trans = torch.tile(cam_transform, (self._num_particles, 1, 1))
            self._particles = self._motion_model(self._particles, tiled_std, tilded_cam_trans)

    @torch.no_grad()
    def systematic_resample_1d(self, w: torch.Tensor) -> torch.Tensor:
        N = w.numel()
        device, dtype = w.device, w.dtype
        s = w.sum()
        if not torch.isfinite(s) or s <= 0:
            return torch.arange(N, device=device)
        p = (w / s).clamp(min=0)
        cdf = torch.cumsum(p, 0); cdf[-1] = 1.0
        u0 = torch.rand((), device=device, dtype=dtype) / N
        u  = u0 + torch.arange(N, device=device, dtype=dtype) / N
        return torch.searchsorted(cdf, u, right=False)
    def update(self, points_2d, ct_points_2d, ct_vis, ctrnet, joint_angles, cam, cTr, gamma, class_result, conf):
        obs_probs, proj_points = self._obs_model(self._particles, points_2d, ct_points_2d, ct_vis, ctrnet, joint_angles, cam, cTr, gamma, class_result, conf)
        self._weights.mul_(obs_probs)
        self._weights.clamp_min_(1e-20)
        self.norm_weights()

        neff = 1./torch.sum(self._weights**2)
        self._since_resample += 1
        if self._prev_joint_angles is not None:
            # did_not_move = torch.any(np.isclose(self._prev_joint_angles, joint_angles))
            if neff < 0.6 * self._num_particles or torch.isnan(neff) or neff > self._num_particles - 10:
                self.resample_hist.append(self._since_resample)
                self._since_resample = 0
                # print("resampling")
                indices = self.systematic_resample_1d(self._weights)
                # print(indices)
                self._particles = self._particles[indices, :]
                self._weights = torch.ones((self._num_particles), dtype=torch.float32).to(device) / torch.tensor(self._num_particles, dtype=torch.float32).to(device)

        self._prev_joint_angles = joint_angles
        return proj_points, self._weights

    def inject_random_particles(self, num_rep):
        std = torch.array([
                1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, # ori
                1.0e-1, 1.0e-1, 1.0e-1, # pos
            ])
        rand_indices = torch.random.randint(0, self._num_particles, size=num_rep)
        prob = torch.sum(self._weights[rand_indices])
        leftover_prob = 1 - prob
        tiled_std = torch.tile(std, (num_rep, 1))
        self._particles[rand_indices, :] = self._init_distribution(tiled_std)
        self._weights[rand_indices] = torch.tile(leftover_prob, (num_rep, 1)).squeeze()/float(num_rep)

    def get_mean_particle(self):
        self.norm_weights()
        return torch.matmul(self._weights, self._particles)

    def get_most_likely_particle(self):
        self.norm_weights()
        idx = int(torch.argmax(self._weights))
        return self._particles[idx]       