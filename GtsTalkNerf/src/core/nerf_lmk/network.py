import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 lmk0=None,
                 R0=None,
                 a0=None,
                 degree_lmk=2,
                 # main network
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=64,
                 num_layers_color=2,
                 hidden_dim_color=64,
                 # deform_ambient net
                 num_layers_ambient=3,
                 hidden_dim_ambient=64,
                 # ambient net
                 ambient_dim=2,
                 ):
        super().__init__(opt, lmk0, R0, a0)
        # ambient network
        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, num_levels=16, level_dim=2,
                                                base_resolution=16, log2_hashmap_size=16,
                                                desired_resolution=2048 * self.bound, interpolation='linear')
        self.encoder_ambient, self.in_dim_ambient = get_encoder('tiledgrid', input_dim=ambient_dim, num_levels=16,
                                                                level_dim=2, base_resolution=16, log2_hashmap_size=16,
                                                                desired_resolution=2048, interpolation='linear')
        self.encoder_lmk, self.in_dim_lmk = get_encoder('landmark', degree=degree_lmk)
        self.encoder_aud, self.in_dim_aud = get_encoder('audio', input_dim=64, degree=0.5)
        self.num_layers_ambient = num_layers_ambient
        self.hidden_dim_ambient = hidden_dim_ambient
        self.ambient_dim = ambient_dim

        self.ambient_net = MLP(self.in_dim + self.in_dim_lmk + self.in_dim_aud, self.ambient_dim, self.hidden_dim_ambient,
                               self.num_layers_ambient)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.sigma_net = MLP(self.in_dim + self.in_dim_ambient, 1 + self.geo_feat_dim, self.hidden_dim,
                             self.num_layers)# + self.num_layers_ambient - 1)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')

        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color,
                             self.num_layers_color)

    def forward(self, x, d, l, R, a, c):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # l: [68, 3]
        # R: [3, 3]
        # a: [64]
        # c: [1, ind_dim], individual code

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        density_out = self.density(x, l, R, a)
        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, density_out['geo_feat'], c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, density_out['geo_feat']], dim=-1)

        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return density_out['sigma'], color, density_out['ambient']

    def density(self, x, l=None, R=None, a=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encoder(x, bound=self.bound)
        if l is None:
            l = self.lmk0
            R = self.R0
            a = self.a0
        #torch.save(dict(x=x,l=l,R=R,p=self.encoder_lmk.state_dict()),'/root/autodl-tmp/dtnerf/tl.pt')
        enc_l = self.encoder_lmk(x, l, R)
        enc_a = self.encoder_aud(a).expand(enc_l.shape[0], -1)

        # ambient
        ambient = torch.cat([enc_x, enc_l, enc_a], dim=1)
        ambient = self.ambient_net(ambient).float()
        ambient = torch.tanh(ambient)  # map to [-1, 1]
        # ambient = torch.zeros_like(enc_x[..., :self.ambient_dim])
        # sigma
        enc_w = self.encoder_ambient(ambient, bound=1)

        h = torch.cat([enc_x, enc_w], dim=-1)
        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient': ambient,
        }

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_ambient.parameters(), 'lr': lr},
            {'params': self.encoder_lmk.parameters(), 'lr': lr},
            {'params': self.ambient_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        return params
