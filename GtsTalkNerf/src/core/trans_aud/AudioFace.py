# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.signal import savgol_filter
from typing import Optional

from .encoders import AudEncoder, lmkEncoder


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(np.log2(n) - 3)))
        return [start ** i for i in range(1, n + 1)]

    if np.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** np.floor(np.log2(n))
        slopes_closest_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
        return slopes_closest_power_of_2 + [j ** 0.5 for j in slopes_closest_power_of_2[::2][:n - closest_power_of_2]]


class CrossMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.):
        super(CrossMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        # self.num_heads = num_heads
        self.dropout = dropout
        # self.head_dim = embed_dim // num_heads
        # assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # nn.init.zeros_(self.q_proj.bias)
        # nn.init.zeros_(self.k_proj.bias)
        # nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(self, query, key, value, **kwargs):
        b, l1, _ = query.shape
        b, l2, _ = key.shape
        assert l1 <= l2
        # if attn_mask is not None and attn_mask.dtype == torch.bool and attn_mask.ndim == 2 and torch.all(
        #         attn_mask == ~torch.eye(*attn_mask.shape).to(attn_mask)):
        o = value[:, :l1]
        if self.training and self.dropout > 0:
            o = F.dropout2d(o.unsqueeze(-1), p=self.dropout).squeeze(-1)
        o = self.o_proj(o)
        return o,


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, emb_dim, nhead, feed_dim, dropout=0.1):
        super().__init__(emb_dim, nhead, feed_dim, dropout=dropout, activation=F.gelu, batch_first=True)
        # self.self_attn = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = CrossMultiheadAttention(emb_dim, dropout=dropout)


class AudioFace(nn.Module):
    def __init__(self,
                 emb_dim: int = 64,
                 nhead: int = 4,
                 feed_dim: int = 512,
                 max_length: int = 256,
                 p: int = 25,
                 dropout: float = 0.1):
        self.config = vars().copy()
        self.config.pop('self')
        self.config.pop('__class__')
        super(AudioFace, self).__init__()
        self.register_buffer("mouthmask", torch.ones(68, dtype=torch.bool))
        self.mouthmask[17:48] = False
        self.max_length = max_length
        x = torch.arange(p).repeat(max_length // p + 1)
        y = x[:, None] * torch.exp(-torch.linspace(0, 1, emb_dim // 2 + 1) * np.log(10000))
        ppe = torch.stack([y, y + torch.pi / 2], -1).view(1, x.size(0), -1)
        self.register_buffer("ppembed", torch.sin(ppe[..., :emb_dim]))

        temmask = torch.full((max_length, max_length), -torch.inf).triu_(diagonal=1)
        for i in range(p, max_length):
            temmask.diagonal(-i).fill_(-(i // p))
        slopes = torch.tensor(get_slopes(nhead))[:, None, None]
        self.register_buffer("tgtmask", (slopes * temmask[None]).float())

        self.register_buffer("memmask", ~torch.eye(max_length).bool())

        self.audenc = AudEncoder(emb_dim)
        self.lmkenc = lmkEncoder(emb_dim)
        self.pidenc = nn.Embedding(9, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            TransformerDecoderLayer(emb_dim, nhead, feed_dim, dropout=dropout),
            2,
        )
        self.mouthproj = nn.Linear(emb_dim, 3 * 37)
        nn.init.zeros_(self.mouthproj.weight)
        nn.init.zeros_(self.mouthproj.bias)

    def forward(self, wav=None, lms=None, formant=None, lmk0=None, pid=None, teacher_forcing=False):
        """
        wav: 1, 640*L
        lms: 1, L, 3, 68
        formant: 1, L, 6
        lmk0: 1, 1, 3, 68
        pid: 1, 1
        """
        rect = [lmk0.amin(3, keepdim=True), lmk0.amax(3, keepdim=True)]
        lim = 0.5
        lmk0 = (lmk0 - rect[0]) / (rect[1] - rect[0]) * 2 * lim - lim
        lms = (lms - rect[0]) / (rect[1] - rect[0]) * 2 * lim - lim
        seq_len = formant.shape[1]
        audfeat = self.audenc(wav, formant)  # 1, L, D
        pidfeat = self.pidenc(pid)  # 1, 1, D
        mouthlmk_0 = lmk0[..., self.mouthmask]  # 1, 1, 3, 37
        mouthlmks = lms[..., self.mouthmask] - mouthlmk_0  # 1, L, 3, 37
        if teacher_forcing:
            lmk_inp = torch.cat([lmk0.new_zeros(1, 1, 3, 37), mouthlmks[:, :-1]], 1)  # 1,L,3,37
            lmkfeat = self.lmkenc(lmk_inp)  # 1, L, D
            lmkfeat = self.dropout(lmkfeat + pidfeat + self.ppembed[:, :seq_len])  # 1, L, D
            tgtmask = self.tgtmask[:, :seq_len, :seq_len]  # nhead, L, L
            memmask = self.memmask[:seq_len, :seq_len]  # L, L
            lmk_pred_feat = self.transformer_decoder(lmkfeat, audfeat, tgt_mask=tgtmask, memory_mask=memmask)
            lmk_pred = self.mouthproj(lmk_pred_feat).view(1, seq_len, 3, -1)
        else:
            lmkfeat = self.lmkenc(lmk0.new_zeros(1, 1, 3, 37))  # 1, 1, D
            lmk_pred = torch.empty_like(mouthlmk_0)
            for i in range(seq_len):
                lmk_emb = self.dropout(lmkfeat + pidfeat + self.ppembed[:, :i + 1])  # 1, i+1, D
                tgtmask = self.tgtmask[:, :i + 1, :i + 1]
                memmask = self.memmask[:i + 1, :seq_len]
                # print(lmk_emb.size(),audfeat.size())
                lmk_pred_feat = self.transformer_decoder(lmk_emb, audfeat, tgt_mask=tgtmask, memory_mask=memmask)
                lmk_pred = self.mouthproj(lmk_pred_feat).view(1, i + 1, 3, -1)  # 1, i+1, 3, 37
                lmkfeat = torch.cat([lmkfeat, self.lmkenc(lmk_pred[:, -1:])], 1)
                # lmkfeat = self.lmkenc(lmk_pred[:, :i+2]);lmkfeat[:, :-1] = lmkfeat[:, :-1].detach()

        mouthloss = F.smooth_l1_loss(lmk_pred, mouthlmks)
        # eyeloss = torch.zeros([])
        return mouthloss, lmk_pred + mouthlmk_0.unsqueeze(0)

    def inference_forward(self, wav=None, formant=None, lmk0=None, pid=None, ret_audio_feat=False, **kwargs):
        rect = [lmk0.amin(3, keepdim=True), lmk0.amax(3, keepdim=True)]
        lim = 0.5
        lmk0 = (lmk0 - rect[0]) / (rect[1] - rect[0]) * 2 * lim - lim
        seq_len = formant.shape[1]
        if seq_len > self.max_length:
            raise ValueError("The required length of landmarks is out of memory")
        audfeat = self.audenc(wav, formant)  # 1, L, D
        pidfeat = self.pidenc(pid)  # 1, 1, D

        lmkfeat = self.lmkenc(lmk0.new_zeros(1, 1, 3, 37))  # 1, 1, D
        lmk_pred = torch.empty(1, seq_len, 3, 68)
        for i in range(seq_len):
            lmk_emb = self.dropout(lmkfeat + pidfeat + self.ppembed[:, :i + 1])  # 1, i+1, D
            tgtmask = self.tgtmask[:, :i + 1, :i + 1]
            memmask = self.memmask[:i + 1, :seq_len]
            lmk_pred_feat = self.transformer_decoder(lmk_emb, audfeat, tgt_mask=tgtmask, memory_mask=memmask)
            lmk_pred = self.mouthproj(lmk_pred_feat).view(1, i + 1, 3, -1)  # 1, i+1, 3, 37
            lmkfeat = torch.cat([lmkfeat, self.lmkenc(lmk_pred[:, -1:])], 1)

        out = lmk0.repeat(1, seq_len, 1, 1)  # 1, L, 3, 68
        out[..., self.mouthmask] += lmk_pred

        eye_wink = nn.init.trunc_normal_(torch.empty(max(seq_len, 10)), mean=0.9, std=0.1, a=0.7, b=1.1)
        eye_wink = savgol_filter(eye_wink.numpy(), 9, 2)

        randrange = (seq_len // 80, seq_len // 50) if seq_len >= 100 else (0, 1)
        winktime = np.random.choice(np.arange(1, max(seq_len // 10, 2)),
                                    np.random.randint(*randrange),
                                    replace=False) * 10
        eye_wink[winktime] = eye_wink[winktime + 8] = 0.6
        eye_wink[winktime + 1] = eye_wink[winktime + 7] = 0.4
        eye_wink[winktime + 2] = eye_wink[winktime + 6] = 0.3
        eye_wink[winktime + 3] = eye_wink[winktime + 5] = 0.1
        eye_wink[winktime + 4] = 0.0
        eye_wink = eye_wink[:seq_len]

        eye_wink = np.expand_dims(eye_wink, axis=(0, *range(2, out.ndim)))
        eye_wink = out.new_tensor(eye_wink)
        out[..., 37:39] = eye_wink * out[..., 37:39] + (1 - eye_wink) * torch.flip(out[..., 40:42], [-1])
        out[..., 43:45] = eye_wink * out[..., 43:45] + (1 - eye_wink) * torch.flip(out[..., 46:48], [-1])

        out = (out + lim) / 2 / lim * (rect[1] - rect[0]) + rect[0]
        if ret_audio_feat:
            return out, audfeat
        else:
            return out
