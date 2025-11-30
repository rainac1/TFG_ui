import math
import numpy as np

import torch
import torch.nn as nn

import raymarching
from .utils import custom_meshgrid, euler_angles_to_matrix


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    import trimesh
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self, opt, lmk0=None, R0=None, a0=None):

        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.density_scale = 1

        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        self.test_train = opt.test_train

        self.cuda_ray = opt.cuda_ray
        lmk0 = torch.rand(68, 3) if lmk0 is None else lmk0
        R0 = torch.eye(3) if R0 is None else R0
        a0 = torch.zeros(64) if a0 is None else a0
        self.register_buffer('lmk0', lmk0)
        self.register_buffer('R0', R0)
        self.register_buffer('a0', a0)
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points,
        # we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound / 2, -opt.bound, opt.bound, opt.bound / 2, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # individual codes
        self.individual_num = opt.ind_num

        self.individual_dim = opt.ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(self.individual_num, self.individual_dim) * 0.1)

        # optimize camera pose
        self.train_camera = self.opt.train_camera
        if self.train_camera:
            self.camera_dR = nn.Parameter(torch.zeros(self.individual_num, 3))  # euler angle
            self.camera_dT = nn.Parameter(torch.zeros(self.individual_num, 3))  # xyz offset

        # extra state for cuda raymarching

        # 3D head density grid
        density_grid = torch.zeros([self.cascade, self.grid_size ** 3])  # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8,
                                       dtype=torch.uint8)  # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0

        # step counter
        step_counter = torch.zeros(16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0

    def init_state(self, lmk0, R0, a0, aabb):
        self.lmk0.data = lmk0.reshape(68, 3).to(self.lmk0)
        self.R0.data = R0.reshape(3, 3).to(self.R0)
        self.a0.data = a0.reshape(-1).to(self.a0)
        self.aabb_train.data = aabb.reshape(6).to(self.aabb_train)
        self.aabb_infer.data = aabb.reshape(6).to(self.aabb_infer)

    def forward(self, x, d, l, R, a, c):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, l=None, R=None):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
            # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run_cuda(self, rays_o, rays_d, lmks, R, auds, index=0, dt_gamma=0, bg_color=None,
                 perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 16]
        # index: [B]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        lmks = lmks.contiguous().view(68, 3)
        R = R.contiguous().view(3, 3)
        auds = auds.contiguous().view(1, -1)

        # only add camera offset at training!
        if self.train_camera and (self.training or self.test_train):
            dT = self.camera_dT[index]  # [1, 3]
            dR = euler_angles_to_matrix(self.camera_dR[index] / 180 * np.pi + 1e-8).squeeze(0)  # [1, 3] --> [3, 3]

            rays_o = rays_o + dT
            rays_d = rays_d @ dR

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        results = {}

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer,
                                                     self.min_near)
        nears = nears.detach()
        fars = fars.detach()

        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[0]
        else:
            ind_code = None

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                                                                    self.cascade, self.grid_size, nears, fars, counter,
                                                                    self.mean_count, perturb, 128, force_all_rays,
                                                                    dt_gamma, max_steps)

            sigmas, rgbs, ambient = self(xyzs, dirs, lmks, R, auds, ind_code)
            sigmas = self.density_scale * sigmas

            # print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights_sum, ambient_sum, depth, image = raymarching.composite_rays_train(
                sigmas, rgbs, ambient.abs().sum(-1), deltas, rays)

            # for training only
            results['weights_sum'] = weights_sum
            results['ambient'] = ambient_sum

        else:

            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                            self.bound, self.density_bitfield, self.cascade,
                                                            self.grid_size, nears, fars, 128,
                                                            perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, ambient = self(xyzs, dirs, lmks, R, auds, ind_code)
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum,
                                           depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # background
        if bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)
        image = image.clamp(0, 1)

        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        results['depth'] = depth
        results['image'] = image

        return results

    @torch.no_grad()
    def mark_untrained_grid(self, cams, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return

        if isinstance(cams, np.ndarray):
            cams = torch.from_numpy(cams)

        B = cams.shape[0]

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        cams = cams.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)
                        mask = (cas_world_xyzs >= self.aabb_infer[:3]) & (cas_world_xyzs <= self.aabb_infer[3:])
                        count[cas, indices] += mask.prod(-1).reshape(-1)

                        # split batch to avoid OOM
#                         head = 0
#                         while head < B:
#                             tail = min(head + S, B)
#                             K, R, T = torch.split(cams[head:tail], [3, 3, 1], dim=-1)
#                             cam_xyzs = torch.baddbmm(T, R, cas_world_xyzs.expand(tail - head, -1, 3).transpose(-1, -2))
#                             fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]

#                             # query if point is covered by any camera
#                             mask_z = cam_xyzs[:, 2, :] > 0  # [S, N]
#                             mask_x = torch.abs(cam_xyzs[:, 0, :]) < cx / fx * cam_xyzs[:, 2, :] + half_grid_size * 2
#                             mask_y = torch.abs(cam_xyzs[:, 1, :]) < cy / fy * cam_xyzs[:, 2, :] + half_grid_size * 2
#                             mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]

#                             # update count
#                             count[cas, indices] += mask
#                             head += S

        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        # print(f'[mark untrained grid] {(count == 0).sum()} from {resolution ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

        # update density grid
        tmp_grid = torch.zeros_like(self.density_grid)

        # full update
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach().to(tmp_grid.dtype)
                        sigmas *= self.density_scale
                        # assign
                        tmp_grid[cas, indices] = sigmas

        # dilate the density_grid (less aggressive culling)
        tmp_grid = raymarching.morton3D_dilation(tmp_grid)

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(
            self.density_grid.clamp(min=0)).item()  # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield.data = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        # update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print('[density grid] min={:.4f}, max={:.4f}, mean={:.4f}, occ_rate={:.3f} | [step counter] mean={}'.format(
        #     self.density_grid.min().item(), self.density_grid.max().item(), self.mean_density,
        #     (self.density_grid > 0.01).sum() / (128**3 * self.cascade), self.mean_count))

    def render(self, rays_o, rays_d, lmks, R, auds, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # auds: [B, 29, 16]
        # eye: [B, 1]
        # bg_coords: [1, N, 2]
        # return: pred_rgb: [B, N, 3]
        results = self.run_cuda(rays_o, rays_d, lmks, R, auds, **kwargs)

        return results
