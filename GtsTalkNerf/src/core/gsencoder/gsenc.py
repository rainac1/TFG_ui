import torch
import torch.nn as nn
import torch.nn.functional as F

from .compute_gauss import compute_gauss, compute_cov3d, trunc_exp
from .freq import FreqEncoder


class GsLmkEncoder(nn.Module):
    def __init__(self,
                 degree: int = 2,
                 n_lmk: int = 68,
                 sigma: float = 0.05):
        super(GsLmkEncoder, self).__init__()
        self.n_lmk = n_lmk
        self._scaling = nn.Parameter(torch.ones(n_lmk, 3) * sigma)
        rots = torch.zeros(n_lmk, 4)
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)
        self.freq = FreqEncoder(n_lmk, degree)
        self.output_dim = self.freq.output_dim

    def forward(self, x: torch.Tensor, l: torch.Tensor, r: torch.Tensor):
        # M,3 68,3 4,4
        wdxyz = x.unsqueeze(0) - l.unsqueeze(1)  # world_dxyz 68, M, 3
        R_t_inverse_col2 = r[:3, 2].clone()
        dz = (wdxyz * R_t_inverse_col2).sum(-1)  # 68, M

        conv3d = compute_cov3d(self._scaling, F.normalize(self._rotation, dim=1))
        w = compute_gauss(wdxyz, conv3d)

        # M = self.build_M(self._scaling, F.normalize(self._rotation, dim=1))
        # conv3d = M @ M.transpose(1, 2)
        # w = (wdxyz @ M).square().sum(2) # w = (wdxyz @ M @ M.transpose(1, 2) @ wdxyz.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
        
        w = trunc_exp(-0.5 * w)
        out = self.freq(dz.T.contiguous())  # (M, k * 68)
        out = (out.view(-1, self.output_dim // self.n_lmk, self.n_lmk) * w.T.unsqueeze(1)).view(-1, self.output_dim)
        return out

    @staticmethod
    @torch.jit.script
    def build_M(s, q):
        S = torch.diag_embed(1 / s)
        R = q.new_zeros(q.shape[0], 3, 3)
        R[:, 0, 0] = 1 - 2 * (q[:, 2]**2 + q[:, 3]**2)
        R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        R[:, 1, 1] = 1 - 2 * (q[:, 1]**2 + q[:, 3]**2)
        R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        R[:, 2, 2] = 1 - 2 * (q[:, 1]**2 + q[:, 2]**2)
        M = R @ S
        return M
